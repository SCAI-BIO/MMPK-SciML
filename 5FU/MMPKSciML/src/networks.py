import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# ========================================
# =========== FROM LHM CODE ==============
# ========================================
class GaussianReparam:
    """Independent Gaussian posterior with re-parameterization trick."""

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def log_density(mu, log_var, z):
        n = dist.normal.Normal(mu, torch.exp(0.5 * log_var))
        log_p = torch.sum(n.log_prob(z), dim=-1)
        return log_p

def get_act(act):
    if act == 'tanh':
        act = nn.Tanh()
    elif act == 'relu':
        act = nn.ReLU()
    elif act == 'selu':
        act = nn.SELU()
    elif act == 'softplus':
        act = nn.Softplus()
    elif act == 'sigmoid':
        act = nn.Sigmoid()
    else:
        act = nn.Identity()
    return act

def get_norm(norm, num_feat):
    if norm == 'instance':
        norm = nn.InstanceNorm1d(num_feat)
    elif norm == 'batch':
        norm = nn.BatchNorm1d(num_feat)
    else:
        norm = nn.Identity()
    return norm

class MLP_ODE(nn.Module):
    def __init__(self, config):
        super(MLP_ODE, self).__init__()

        latent_dim = config.h_odesize
        input_dim = config.i_ode + 1 # Centr compartment
        out_dim = config.dim_params
        num_layers = config.odenet_n_layers
        norm_name = config.norm_ode
        act_name = config.act_odenet

        act = get_act(act_name)

        layers = []
        for i in range(num_layers):

            if num_layers == 1:
                layers.append(nn.Linear(input_dim, out_dim))
            elif i == 0:
                layers.append(nn.Linear(input_dim, latent_dim))
                layers.append(get_norm(norm_name, latent_dim))
            elif i == num_layers -1:
                layers.append(nn.Linear(latent_dim, out_dim))
            else:
                layers.append(nn.Linear(latent_dim, latent_dim))
                layers.append(get_norm(norm_name, latent_dim))

            if i < num_layers -1:
                layers.append(act)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class BottleNeckEnc(nn.Module, GaussianReparam):
    # Latent ODE with bottleneck structure
    # num_odelayers specifies the depth of the Neural ODE
    def __init__(self, config):
        super (BottleNeckEnc, self).__init__()

        latent_dim = config.hh_ssize
        input_dim = config.i_ssize + config.i_tsize
        out_dim = config.dim_params
        num_layers = config.senc_n_layers
        norm_name = config.norm_senc
        act_name = config.act_senc
        l_act_name = config.l_act_senc

        act = get_act(act_name)
        l_act = get_act(l_act_name)

        diff = latent_dim - input_dim

        assert num_layers > 1, 'You need at least 2 layers'

        layers_enc = []
        layers_dec = []

        # Encoder and decoder parts of the ODE function
        # input and output are just the opposite
        # at the end we reverse the dec list to be aligned
        # with the encoder one
        for i in range(num_layers):

            fact_i = i/num_layers
            fact_o = (i+1)/num_layers

            c_i = input_dim + int(np.round(diff*fact_i))
            c_o = input_dim + int(np.round(diff*fact_o))

            if fact_o == 1:
                layers_enc.append(nn.Linear(c_i, latent_dim))
                layers_enc.append(get_norm(norm_name, latent_dim))
                layers_dec.append(act)
                layers_dec.append(get_norm(norm_name, c_i))
                layers_dec.append(nn.Linear(latent_dim, c_i))

            else:
                layers_enc.append(nn.Linear(c_i, c_o))
                layers_enc.append(get_norm(norm_name, c_o))

                if i == 0:
                    layers_dec.append(l_act)
                    # layers_dec.append(nn.Linear(c_o, out_dim))
                    layers_dec.append(nn.Linear(c_o, c_i))
                else:
                    layers_dec.append(act)
                    layers_dec.append(get_norm(norm_name, c_i))
                    layers_dec.append(nn.Linear(c_o, c_i))
                # layers_dec.append(get_norm(norm_name, c_i))

            layers_enc.append(act)
            
        layers_enc.append(nn.Linear(latent_dim, latent_dim))
        layers_enc.append(get_norm(norm_name, latent_dim))
        layers_enc.append(act)

        layers_dec.reverse()
        layers = layers_enc + layers_dec  # concat enc + dec

        self.model = nn.Sequential(*layers)

        self.act_mean = get_act(config.act_mean)
        self.act_var = get_act(config.act_var)

        mean, var = [], []
        for i in range(config.mlp_n_layers):

            # The last layer does not have an activation
            if config.mlp_n_layers == 1 or i == (config.mlp_n_layers - 1):
                mean.append(nn.Linear(c_i, config.dim_params))
                var.append(nn.Linear(c_i, config.dim_params))
            else:
                mean.append(nn.Linear(c_i, c_i))
                mean.append(self.act_mean)
                var.append(nn.Linear(c_i, c_i))
                var.append(self.act_var)

        self.mean = nn.Sequential(*mean)
        self.var = nn.Sequential(*var)

    def forward(self, x):
        out = self.model(x)

        mean = self.act_mean(self.mean(out))
        var = self.act_var(self.var(out))

        return out, mean, var


class MLP_Enc(nn.Module, GaussianReparam):

    def __init__(self, config):
        super (MLP_Enc, self).__init__()

        latent_dim = config.hh_ssize
        input_dim = config.i_ssize + config.i_tsize
        out_dim = config.dim_params
        num_layers = config.senc_n_layers
        norm_name = config.norm_senc
        act_name = config.act_senc
        l_act_name = config.l_act_senc

        act = get_act(act_name)
        l_act = get_act(l_act_name)

        layers = []
        for i in range(num_layers):

            if num_layers == 1:
                layers.append(nn.Linear(input_dim, out_dim))
            elif i == 0:
                layers.append(nn.Linear(input_dim, latent_dim))
                layers.append(get_norm(norm_name, latent_dim))
            else:
                layers.append(nn.Linear(latent_dim, latent_dim))
                layers.append(get_norm(norm_name, latent_dim))

            if i < num_layers -1:
                layers.append(act)
            else:
                layers.append(l_act)

        self.model = nn.Sequential(*layers)

        self.act_mean = get_act(config.act_mean)
        self.act_var = get_act(config.act_var)

        mean, var = [], []
        for i in range(config.mlp_n_layers):

            # The last layer does not have an activation
            if config.mlp_n_layers == 1 or i == (config.mlp_n_layers - 1):
                mean.append(nn.Linear(latent_dim, config.dim_params))
                var.append(nn.Linear(latent_dim, config.dim_params))
            else:
                mean.append(nn.Linear(latent_dim, latent_dim))
                mean.append(self.act_mean)
                var.append(nn.Linear(latent_dim, latent_dim))
                var.append(self.act_var)

        self.mean = nn.Sequential(*mean)
        self.var = nn.Sequential(*var)

    def forward(self, x):
        out = self.model(x)

        mean = self.act_mean(self.mean(out))
        var = self.act_var(self.var(out))

        return out, mean, var


# ========================================
# ================== ODE =================
# ========================================
class ODEFunc_PPCL_PopV(nn.Module):
    def __init__(self, config, IC=[20.0, 40.0]):
        super (ODEFunc_PPCL_PopV, self).__init__()

        # ODE system defined in hours
        self.days_inf = config.infusion_time/24
        self.i_time = config.infusion_time # hours
        if config.fix_V:
            self.V_pop = torch.log(torch.tensor(config.IC_V))
        else:
            self.V_pop = nn.Parameter(torch.log(torch.tensor(config.IC_V)))

        self.CL_pop = nn.Parameter(torch.log(torch.tensor(IC[0])))

    def update_params(self, params):
        self.CL = torch.exp(params[:, :1]) * torch.exp(self.CL_pop)
        self.V = torch.exp(self.V_pop)

    def update_dose(self, dose):
        self.Dose = dose

    def forward(self, t, u):
        # ODE system defined in hours using t=1 as 1 week

        # AUC = [mg * h / L] 
        # Doses = [mg]
        # CL = L/h
        # dCDt = mg/h  - ([L/(h*L)] * [mg]
        # then when dividing dCdt the concentration becomes
        # Conc = mg/L

        Centr = u[:, :1]
        dCdt = (self.Dose/self.i_time).unsqueeze(-1) - (self.CL/self.V) * Centr
        return dCdt


class ODEFunc_CLV_PP(nn.Module):
    def __init__(self, config, IC=[20.0, 40.0]):
        super (ODEFunc_CLV_PP, self).__init__()

        # ODE system defined in hours
        self.days_inf = config.infusion_time/24
        self.i_time = config.infusion_time # hours
        self.CL_pop = nn.Parameter(torch.log(torch.tensor(IC[0])))
        self.V_pop = nn.Parameter(torch.log(torch.tensor(IC[1])))

    def update_params(self, params):
        self.CL = torch.exp(params[:, :1]) * torch.exp(self.CL_pop)
        self.V = torch.exp(params[:, 1:]) * torch.exp(self.V_pop)

    def update_dose(self, dose):
        self.Dose = dose

    def forward(self, t, u):
        # ODE system defined in hours using t=1 as 1 week

        # AUC = [mg * h / L] 
        # Doses = [mg] 
        # CL = L/h
        # dCDt = mg/h  - ([L/(h*L)] * [mg]
        # then when dividing dCdt the concentration becomes
        # Conc = mg/L

        Centr = u[:, :1]
        dCdt = (self.Dose/self.i_time).unsqueeze(-1) - (self.CL/self.V) * Centr
        return dCdt

