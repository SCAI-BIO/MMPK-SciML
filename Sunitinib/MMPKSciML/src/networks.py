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
    elif norm == 'layer':
        norm = nn.LayerNorm(num_feat)
    else:
        norm = nn.Identity()
    return norm

# ========================================
# ============ IMPUTATION ================
# ========================================
class VaderLayer(nn.Module):
    def __init__(self, A_init):
        super(VaderLayer, self).__init__()
        self.b = nn.Parameter(A_init)

    def forward(self, X, W):
        # X is the data and w is the indicator function
        # Handle missing values section of the main text
        return ((1.0 - W) * self.b + X * W).float()

# ========================================
# =========== STATIC ENCODER =============
# ========================================
class MLP_Enc(nn.Module):

    def __init__(self, config):
        super (MLP_Enc, self).__init__()

        input_dim = config.i_ssize
        out_dim = config.senc_h_size
        num_layers = config.senc_n_layers
        norm_name = config.norm_senc
        act_name = config.act_senc

        act = get_act(act_name)
        layers = []
        for i in range(num_layers):

            if num_layers == 1:
                layers.append(nn.Linear(input_dim, out_dim))
            elif i == 0:
                layers.append(nn.Linear(input_dim, out_dim))
                layers.append(get_norm(norm_name, out_dim))
            elif i == num_layers -1:
                # Last layer does not have norm nor activation
                layers.append(nn.Linear(out_dim, out_dim))
            else:
                layers.append(nn.Linear(out_dim, out_dim))
                layers.append(get_norm(norm_name, out_dim))

            if i < num_layers -1:
                layers.append(act)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

# ========================================
# ============ TLSTM  ENCODER ============
# ========================================
class TLSTMCell(nn.Module):
    def __init__(self, i_size, h_size):
        super(TLSTMCell, self).__init__()

        self.layers = nn.Linear(i_size + h_size, 4 * h_size)
        self.desc = nn.Linear(h_size, h_size)

        self.h_size = h_size

        self.c1 = torch.tensor(1)
        self.c2 = torch.tensor(2.7183)

    def map_elapse_time(self, t):
        T = self.c1 / torch.log(t + self.c2)
        # Ones = torch.ones(1, self.h_size, dtype=T.dtype).to(T.device) # .to(self.c1.device)
        Ones = torch.ones(1, self.h_size).to(T.device) # .to(self.c1.device)
        T = torch.matmul(T, Ones)
        return T 

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        x, dt = input_tensor[:, :-1], input_tensor[:, -1:]
        # Map elapse time in days or months
        T = self.map_elapse_time(dt)
        
        # Decompose the previous cell if there is a elapse time
        C_ST = torch.tanh(self.desc(h_cur))
        C_ST_dis = T * C_ST
        # if T is 0, then the weight is one
        h_cur = h_cur - C_ST + C_ST_dis

        combined = torch.cat((x, h_cur), 1)
        combined_out = self.layers(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_out, self.h_size, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class TLSTM_Enc(nn.Module):
    def __init__(self, config):
        super(TLSTM_Enc, self).__init__()

        # DiffT is not a feature for the LSTM
        i_size = config.i_tsize - 1
        self.h_size = config.lstm_h_size
        self.n_layers = config.lstm_n_layers

        cell_list = []
        for i in range(self.n_layers):
            c_inp = i_size if i == 0 else self.h_size
            cell_list.append(TLSTMCell(c_inp, self.h_size))
        
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, inputs_t):

        inputs, t = inputs_t[..., :-1], inputs_t[..., -1:]
        b, seq_l, _ = inputs.size()

        for l_idx in range(self.n_layers):

            h = torch.zeros(b, self.h_size).to(inputs.device)
            c = torch.zeros(b, self.h_size).to(inputs.device)

            if l_idx == 0:
                curr_inp = inputs_t.clone()

            output_inner = []
            for t_seq in reversed(range(seq_l)):
                h, c = self.cell_list[l_idx](curr_inp[:, t_seq], [h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            curr_inp = torch.cat((layer_output, t), -1)

        return h


# ========================================
# ============= FULL ENCODER =============
# ========================================
class Encoder(nn.Module, GaussianReparam):
    def __init__(self, config, AT, AS):
        super(Encoder, self).__init__()

        self.Time_ImpLayer = VaderLayer(AT)
        self.Time_Encoder = TLSTM_Enc(config)

        self.Static_ImpLayer = VaderLayer(AS)
        self.Static_Encoder = MLP_Enc(config)

        self.h_size = config.lstm_h_size + config.senc_h_size
        self.mlp_layers = config.mlp_n_layers
        self.act_mean = get_act(config.act_mean)
        self.act_var = get_act(config.act_var)
        norm_mean = config.norm_mean
        norm_var = config.norm_var

        out_mlp = config.dim_params
        mean, var = [], []
        for i in range(self.mlp_layers):

            # The last layer does not have an activation
            if self.mlp_layers == 1 or i == (self.mlp_layers - 1):
                mean.append(nn.Linear(self.h_size, out_mlp))
                var.append(nn.Linear(self.h_size, out_mlp))
            else:
                mean.append(nn.Linear(self.h_size, self.h_size))
                mean.append(get_norm(norm_mean, self.h_size))
                if config.act_mean != 'none':
                    mean.append(self.act_mean)
                var.append(nn.Linear(self.h_size, self.h_size))
                var.append(get_norm(norm_var, self.h_size))
                if config.act_var != 'none':
                    var.append(self.act_var)

        self.mean = nn.Sequential(*mean)
        self.var = nn.Sequential(*var)

    def forward(self, COVL, MCOVL, COVS, MCOVS):

        COVL = self.Time_ImpLayer(COVL, MCOVL)
        t_rep = self.Time_Encoder(COVL)

        COVS = self.Static_ImpLayer(COVS, MCOVS)
        stat_rep = self.Static_Encoder(COVS)

        rep = torch.cat((stat_rep, t_rep), 1)
        mean = self.act_mean(self.mean(rep))
        var = self.act_var(self.var(rep))

        return rep, mean, var

# ========================================
# ============== Math ODEs ===============
# ========================================
class Math_ODE(nn.Module):
    def __init__(self, config):
        super (Math_ODE, self).__init__()

        if config.fix_KA:
            self.pop_KA = torch.log(torch.tensor(config.IC_KA))
        else:
            self.pop_KA = nn.Parameter(torch.log(torch.tensor(config.IC_KA)))

        self.QH = torch.tensor(config.IC_QH)
        self.pop_F_Met = torch.tensor(config.IC_F_Met)

        self.pop_CL_S = nn.Parameter(torch.log(torch.tensor(config.IC_CL_S)))
        self.pop_Q_S = nn.Parameter(torch.log(torch.tensor(config.IC_Q_S)))
        self.pop_V2_S = nn.Parameter(torch.log(torch.tensor(config.IC_V2_S)))
        self.pop_V3_S = torch.tensor(config.IC_V3_S)

        self.pop_CL_Met = nn.Parameter(torch.log(torch.tensor(config.IC_CL_Met)))
        self.pop_Q_Met = nn.Parameter(torch.log(torch.tensor(config.IC_Q_Met)))
        self.pop_V2_Met = nn.Parameter(torch.log(torch.tensor(config.IC_V2_Met)))
        self.pop_V3_Met = nn.Parameter(torch.log(torch.tensor(config.IC_V3_Met)))

        self.t_ode = config.type_ode

    def update_params(self, params):

        self.KA = torch.exp(self.pop_KA)
        # Eta values only for CL_S, V2_S, V2_Met and F_Met
        self.CL_S = torch.exp(self.pop_CL_S) * torch.exp(params[:, :1])
        n_pat = len(self.CL_S)
        self.Q_S = torch.exp(self.pop_Q_S).repeat(n_pat).unsqueeze(-1)
        self.V2_S = torch.exp(self.pop_V2_S) * torch.exp(params[:, 1:2])
        self.V3_S = self.pop_V3_S.repeat(n_pat).unsqueeze(-1)

        self.CL_Met = torch.exp(self.pop_CL_Met).repeat(n_pat).unsqueeze(-1)
        self.Q_Met = torch.exp(self.pop_Q_Met).repeat(n_pat).unsqueeze(-1)
        self.V2_Met = torch.exp(self.pop_V2_Met) * torch.exp(params[:, 2:3])
        self.V3_Met = torch.exp(self.pop_V3_Met).repeat(n_pat).unsqueeze(-1)
        self.F_Met = self.pop_F_Met * torch.exp(params[:, 3:4])

        self.QH_ = self.QH.repeat(n_pat).unsqueeze(-1)
        self.KH = self.QH_ / self.V2_S 

        self.K23 = self.Q_S / self.V2_S
        self.K32 = self.Q_S / self.V3_S
        self.K45 = self.Q_Met / self.V2_Met
        self.K54 = self.Q_Met / self.V3_Met
        self.KE_Met = self.CL_Met / self.V2_Met

        self.ode_params = torch.cat((
            self.CL_S, self.Q_S, self.V2_S, self.V3_S,
            self.CL_Met, self.Q_Met, self.V2_Met,
            self.V3_Met, self.F_Met), 1)

    def forward(self, t, u):

        Depot = u[:, :1]
        CentrS = u[:, 1:2]
        C3S = u[:, 2:3]
        CentrM = u[:, 3:4]
        C3M = u[:, 4:5]

        CLIV = (self.KA* Depot + self.KH*CentrS) / (self.QH_ + self.CL_S)
        dDdt = - self.KA*Depot
        dCSdt = self.QH_*CLIV - self.KH*CentrS - self.K23*CentrS + self.K32*C3S
        dC3Sdt = self.K23*CentrS - self.K32*C3S
        dCMdt = (self.F_Met*self.CL_S)*CLIV - self.KE_Met*CentrM - self.K45*CentrM + self.K54*C3M
        dC3Mdt = self.K45*CentrM - self.K54*C3M

        return torch.cat([dDdt, dCSdt, dC3Sdt, dCMdt, dC3Mdt], dim=-1)


class MathW_ODE(nn.Module):
    def __init__(self, config):
        super (MathW_ODE, self).__init__()

        if config.fix_KA:
            self.pop_KA = torch.log(torch.tensor(config.IC_KA))
        else:
            self.pop_KA = nn.Parameter(torch.log(torch.tensor(config.IC_KA)))

        self.QH = torch.tensor(config.IC_QH)
        self.pop_F_Met = torch.tensor(config.IC_F_Met)

        self.pop_CL_S = nn.Parameter(torch.log(torch.tensor(config.IC_CL_S)))
        self.pop_Q_S = nn.Parameter(torch.log(torch.tensor(config.IC_Q_S)))
        self.pop_V2_S = nn.Parameter(torch.log(torch.tensor(config.IC_V2_S)))
        self.pop_V3_S = torch.tensor(config.IC_V3_S)

        self.pop_CL_Met = nn.Parameter(torch.log(torch.tensor(config.IC_CL_Met)))
        self.pop_Q_Met = nn.Parameter(torch.log(torch.tensor(config.IC_Q_Met)))
        self.pop_V2_Met = nn.Parameter(torch.log(torch.tensor(config.IC_V2_Met)))
        self.pop_V3_Met = nn.Parameter(torch.log(torch.tensor(config.IC_V3_Met)))

        self.t_ode = config.type_ode
        self.log_scale = config.log_scale

    def update_params(self, params, weight):

        if self.log_scale:
            weight = torch.exp(weight)

        ASCL = (weight/70)**0.75
        ASV = (weight/70)
        self.KA = torch.exp(self.pop_KA)
        # Eta values only for CL_S, V2_S, V2_Met and F_Met
        self.CL_S = torch.exp(self.pop_CL_S) * torch.exp(params[:, :1]) * ASCL
        n_pat = len(self.CL_S)
        self.Q_S = torch.exp(self.pop_Q_S).repeat(n_pat).unsqueeze(-1) * ASCL
        self.V2_S = torch.exp(self.pop_V2_S) * torch.exp(params[:, 1:2]) * ASV
        self.V3_S = self.pop_V3_S.repeat(n_pat).unsqueeze(-1) * ASV

        self.CL_Met = torch.exp(self.pop_CL_Met).repeat(n_pat).unsqueeze(-1) * ASCL
        self.Q_Met = torch.exp(self.pop_Q_Met).repeat(n_pat).unsqueeze(-1) * ASCL
        self.V2_Met = torch.exp(self.pop_V2_Met) * torch.exp(params[:, 2:3]) * ASV
        self.V3_Met = torch.exp(self.pop_V3_Met).repeat(n_pat).unsqueeze(-1) * ASV
        self.F_Met = self.pop_F_Met * torch.exp(params[:, 3:4])

        self.QH_ = self.QH.repeat(n_pat).unsqueeze(-1) * ASCL
        self.KH = self.QH_ / self.V2_S 

        self.K23 = self.Q_S / self.V2_S
        self.K32 = self.Q_S / self.V3_S
        self.K45 = self.Q_Met / self.V2_Met
        self.K54 = self.Q_Met / self.V3_Met
        self.KE_Met = self.CL_Met / self.V2_Met

        self.ode_params = torch.cat((
            self.CL_S, self.Q_S, self.V2_S, self.V3_S,
            self.CL_Met, self.Q_Met, self.V2_Met,
            self.V3_Met, self.F_Met), 1)

    def forward(self, t, u):

        Depot = u[:, :1]
        CentrS = u[:, 1:2]
        C3S = u[:, 2:3]
        CentrM = u[:, 3:4]
        C3M = u[:, 4:5]

        CLIV = (self.KA* Depot + self.KH*CentrS) / (self.QH_ + self.CL_S)
        dDdt = - self.KA*Depot
        dCSdt = self.QH_*CLIV - self.KH*CentrS - self.K23*CentrS + self.K32*C3S
        dC3Sdt = self.K23*CentrS - self.K32*C3S
        dCMdt = (self.F_Met*self.CL_S)*CLIV - self.KE_Met*CentrM - self.K45*CentrM + self.K54*C3M
        dC3Mdt = self.K45*CentrM - self.K54*C3M

        return torch.cat([dDdt, dCSdt, dC3Sdt, dCMdt, dC3Mdt], dim=-1)