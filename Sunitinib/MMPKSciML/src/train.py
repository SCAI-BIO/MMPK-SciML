import os
import warnings
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
from termcolor import colored
from glob import glob
import torch.distributions as dist
from networks import *
from utils import *
from visualizations import *
from solver import Solver
warnings.filterwarnings('ignore')


class Train(Solver):
    def __init__(self, config, dataloader, dataloader_val):
        super(Train, self).__init__(config, dataloader, dataloader_val)

        if config.mode == 'val':
            self.validate()
            print('Validation has finished')
        else:
            self.run()
            print('Training has finished')

    @torch.no_grad()
    def save_fig_train(self, data, epoch):

        save_path = os.path.join(self.config.save_path_samples,'Train_Imgs')
        os.makedirs(save_path, exist_ok=True)

        Conc = data[0]        # Batch x Max_Num_Mea x 1
        COVL = data[1]        # Batch x Max_Num_Mea x DL
        MCOVL = data[2]       # Batch x Max_Num_Mea x DL
        COVS = data[3]        # Batch x Ds
        MCOVS = data[4]       # Batch x Ds
        Doses_ODE = data[6]   # Batch x Time_ODE x 1
        Mask_ODE = data[7]    # Batch x Time_ODE
        Time_ODE = data[8]    # Time_ODE
        Time_DV = data[9]     # Batch x Max_Num_Mea

        Tsld = COVL[..., -2:-1][..., 0]

        b = Conc.size(0)
        _, mean, log_var = self.Enc(COVL, MCOVL, COVS, MCOVS)
        etas = self.Enc.reparameterize(mean, log_var)
        # etas = torch.zeros_like(mean).to(self.device)
        CI = None

        if self.config.type_ode == 'MathW_ODE':
            COVS_ = self.Enc.Static_ImpLayer(COVS, MCOVS)
            self.ODE.update_params(etas, COVS_[:, 2:3])
        else:
            self.ODE.update_params(etas)
        pred_z, pred_full = self.run_ODE(
            b, Doses_ODE, Time_ODE,
            Mask_ODE, CI=CI, full=True)
        Conc = Conc[:, :, 0]

        # It could be that the size of the examples for the training
        # fig is shorterthat the one of the full dataset
        s_batch = pred_z.size(1)
        Conc = Conc[..., :s_batch]
        Tsld = Tsld[..., :s_batch]
        Time_DV = Time_DV[..., :s_batch]

        Conc = Conc.detach().cpu()
        pred_z = pred_z.detach().cpu()
        pred_full = pred_full.detach().cpu()
        Time_DV = Time_DV.detach().cpu()
        Time_ODE = Time_ODE.cpu()
        if self.config.norm_T:
            Time_DV = denormalize(Time_DV, self.config.Tmin, self.config.Tmax)
            Time_ODE = denormalize(Time_ODE, self.config.Tmin, self.config.Tmax)

        save_name = os.path.join(save_path,'Epoch_%d.png'%(epoch))
        diffsamp_mean_traj(
            Conc.unsqueeze(1), pred_full.unsqueeze(1), Time_DV, Time_ODE,
            epoch, save_name)

        save_name = os.path.join(save_path,'Scatter_Epoch_%d.png'%(epoch))
        scatter_dcolors(Conc, pred_z, save_name, name='Conc')
        save_name = os.path.join(save_path,'Log_Scatter_Epoch_%d.png'%(epoch))
        scatter_dcolors(Conc, pred_z, save_name, name='Conc', log=True)
        save_name = os.path.join(save_path,'Error_Epoch_%d.png'%(epoch))
        # GOF_plots(Conc, pred_z, Time_DV, Tsld, epoch, save_name)
        Residual_plots(Conc, pred_z, Time_DV, Tsld, epoch, save_name)
        self.save_ode_params(save_path, epoch)

    @torch.no_grad()
    def get_pred_val(self, data, gname, epoch, save_path, save_fig=False):

        if save_fig:
            save_path = os.path.join(save_path, gname)
            os.makedirs(save_path, exist_ok=True)

        Conc = data[0][0].to(self.device)        # Batch x Max_Num_Mea x 1
        COVL = data[1][0].to(self.device)        # Batch x Max_Num_Mea x DL
        MCOVL = data[2][0].to(self.device)       # Batch x Max_Num_Mea x DL
        COVS = data[3][0].to(self.device)        # Batch x Ds
        MCOVS = data[4][0].to(self.device)       # Batch x Ds
        Mask_Loss = data[5][0, :, :, 0].to(self.device)     # Batch x Time x 1
        Doses_ODE = data[6][0].to(self.device)   # Batch x Time_ODE x 1
        Mask_ODE = data[7][0].to(self.device)    # Batch x Time_ODE
        Time_ODE = data[8][0].to(self.device)    # Time_ODE
        Time_DV = data[9][0, :, :, 0].to(self.device)    # Time_ODE

        Doses = COVL[..., 4:5]
        Tsld = COVL[..., -2:-1]
        COVS_ = self.Enc.Static_ImpLayer(COVS, MCOVS)
        COVL_ = self.Enc.Time_ImpLayer(COVL, MCOVL)
        Met = COVL_[..., 1:2]

        b = Conc.size(0)
        Conc = Conc[..., 0]
        Tsld = Tsld[:, :, 0]
        Weight = COVS_[:, 2:3]

        total_real = list()
        total_pred = list()
        total_pred_full = list()
        total_params = list()
        total_etas = list()
        total_pred_met = list()

        _, eta_means, eta_log_vars = self.Enc(COVL, MCOVL, COVS, MCOVS)
        patient_dist = dist.normal.Normal(eta_means, torch.exp(0.5*eta_log_vars))
        if 'Prior' in self.config.val_data_split:
            p_mean = torch.ones_like(eta_means).to(self.device) * self.config.prior_mean
            p_std = torch.ones_like(eta_means).to(self.device) * self.config.prior_std
            patient_dist = dist.normal.Normal(p_mean, p_std)

        for nrun in range(self.config.nruns_ppd):

            if 'PostDist' in self.config.val_data_split:
                val_etas = []
                for _ in range(b):
                    val_etas.append(self.Mean_Dists.sample())
                val_etas = torch.stack(val_etas, 0)
            else:
                val_etas = patient_dist.sample()
            CI = None
            total_etas.append(val_etas.unsqueeze(1))
            if self.config.type_ode == 'MathW_ODE':
                self.ODE.update_params(val_etas, Weight)
            else:
                self.ODE.update_params(val_etas)

            pred_x, pred_x_full = self.run_ODE(b, Doses_ODE, Time_ODE,
                                               Mask_ODE, CI=CI, full=True,
                                               metabolite=True)

            pred_met, pred_x = pred_x[..., 1], pred_x[..., 1]
            total_pred_met.append(pred_met.unsqueeze(1))

            total_real.append(Conc.unsqueeze(1))
            total_pred.append(pred_x.unsqueeze(1))
            total_pred_full.append(pred_x_full.unsqueeze(1))
            ode_params = torch.log(self.ODE.ode_params).unsqueeze(1)
            total_params.append(ode_params)
        
        Conc = torch.cat(total_real, 1)
        pred_x = torch.cat(total_pred, 1)
        pred_x_full = torch.cat(total_pred_full, 1)
        total_params = torch.cat(total_params, 1)
        total_etas = torch.cat(total_etas, 1)
        total_pred_met = torch.cat(total_pred_met, 1)

        # The metrics as well as the GOF plots need to be done with the
        # patient specific mean
        # Calculating the Trajectory with the Eta mean
        if self.config.type_ode == 'MathW_ODE':
            self.ODE.update_params(eta_means, COVS_[:, 2:3])
        else:
            self.ODE.update_params(eta_means)
        CI = None
        GOF_SEQ = self.run_ODE(b, Doses_ODE, Time_ODE,
                               Mask_ODE, CI=CI,
                               metabolite=True)
        GOF_MET, GOF_SEQ = GOF_SEQ[..., 1], GOF_SEQ[..., 0]

        # Calculating the PRED           
        # The Etas need to be 0
        PRED_etas = torch.zeros_like(val_etas).to(self.device)
        if self.config.type_ode == 'MathW_ODE':
            self.ODE.update_params(PRED_etas, COVS_[:, 2:3])
        else:
            self.ODE.update_params(PRED_etas)
        CI = None
        PRED = self.run_ODE(b, Doses_ODE, Time_ODE,
                            Mask_ODE, CI=CI)

        tt = Conc.shape[-1]
        x_m = Conc.clone().detach().float().view(-1, tt)
        pred_m = pred_x.clone().detach().float().view(-1, tt)
        Mask_Loss = Mask_Loss.repeat(self.config.nruns_ppd, 1)
        mape, smape, rmse = mape_smape_diffsamp(x_m, pred_m, Mask_Loss)

        Conc = Conc.detach().cpu()
        pred_x = pred_x.detach().cpu()
        pred_x_full = pred_x_full.detach().cpu()
        PRED = PRED.detach().cpu()
        GOF_SEQ = GOF_SEQ.detach().cpu()
        Time_DV = Time_DV.detach().cpu()
        Time_ODE = Time_ODE.cpu()
        if self.config.norm_T:
            Time_DV = denormalize(Time_DV, self.config.Tmin, self.config.Tmax)
            Time_ODE = denormalize(Time_ODE, self.config.Tmin, self.config.Tmax)

        if save_fig:
            save_name_d = os.path.join(save_path,'Val_Epoch%d.png'%(epoch))
            save_name_gof = os.path.join(save_path,'Val_Epoch%d_GOF.png'%(epoch))

            pats = Conc.shape[0]
            if pats > 10:
                pats = 10

            Conc_fig = Conc[: pats].clone()
            Pred_x_fig = pred_x[: pats].clone()
            Pred_x_full_fig = pred_x_full[: pats].clone()
            Time_DV_fig = Time_DV[: pats].clone()

            diffsamp_mean_traj(
                Conc_fig, Pred_x_full_fig, Time_DV_fig, Time_ODE,
                epoch, save_name_d)
            # GOF_plots(Conc, pred_x, Time_DV, epoch, save_name_gof, val=True) 

            Conc_fig = Conc_fig[:, 0]
            Pred_x_fig = Pred_x_fig[:, 0]

            Conc_fig = Conc_fig.view(pats, -1)
            Pred_x_fig = Pred_x_fig.view(pats, -1)        
            Time_DV_fig = Time_DV_fig.view(pats, -1)
            Tsld_fig = Tsld[: pats].clone()

            # save_name = os.path.join(save_path,'Scatter_Epoch_%d_%s.png'%(epoch, gname))
            # scatter_dcolors(Conc_fig, Pred_x_fig, save_name, name='Conc')
            save_name = os.path.join(save_path,'Log_Scatter_Epoch_%d_%s.png'%(epoch, gname))
            scatter_dcolors(Conc_fig, Pred_x_fig, save_name, name='Conc', log=True)
            save_name = os.path.join(save_path,'Error_Epoch_%d_%s.png'%(epoch, gname))
            Residual_plots(Conc_fig, Pred_x_fig, Time_DV_fig, Tsld_fig, epoch, save_name)

        return (Conc, pred_x, pred_x_full, PRED, GOF_SEQ, Met, 
                total_pred_met, GOF_MET, Mask_Loss, eta_means,
                eta_log_vars, total_etas, total_params, Weight,
                Doses, Tsld, Time_DV, Doses_ODE, Mask_ODE,
                Time_ODE, mape, smape, rmse)

    @torch.no_grad()
    def save_ode_params(self, save_path, epoch):

        save_path = os.path.join(save_path, 'ODE_Params.csv')

        KA = round(torch.exp(self.ODE.pop_KA).item(), 3)
        QH = round(self.ODE.QH.item(), 3)

        CL_S = round(torch.exp(self.ODE.pop_CL_S).item(), 3)
        Q_S = round(torch.exp(self.ODE.pop_Q_S).item(), 3)
        V2_S = round(torch.exp(self.ODE.pop_V2_S).item(), 3)
        V3_S = round(self.ODE.pop_V3_S.item(), 3)

        CL_M = round(torch.exp(self.ODE.pop_CL_Met).item(), 3)
        Q_M = round(torch.exp(self.ODE.pop_Q_Met).item(), 3)
        V2_M = round(torch.exp(self.ODE.pop_V2_Met).item(), 3)
        V3_M = round(torch.exp(self.ODE.pop_V3_Met).item(), 3)
        F_M = round(self.ODE.pop_F_Met.item(), 3)

        print('KA: ', KA)
        print('QH: ', QH)
        print('CL_S: ', CL_S)
        print('Q_S: ', Q_S)
        print('V2_S: ', V2_S)
        print('V3_S: ', V3_S)
        print('CL_M: ', CL_M)
        print('Q_M: ', Q_M)
        print('V2_M: ', V2_M)
        print('V3_M: ', V3_M)
        print('F_M: ', F_M)

        header = ['Epoch', 'KA', 'QH', 'CL_S', 'Q_S', 'V2_S', 'V3_S',
                  'CL_M', 'Q_M', 'V2_M', 'V3_M', 'F_M']
        data = [epoch, KA, QH, CL_S, Q_S, V2_S, V3_S,
                CL_M, Q_M, V2_M, V3_M, F_M]

        import csv
        typ = 'a' if os.path.exists(save_path) else 'wt'
        
        with open(save_path, typ, encoding='UTF8') as file:

            writer = csv.writer(file)
            if typ == 'wt':
                writer.writerow(header)
            writer.writerow(data)

    def run_ODE(self, b, Doses_ODE, Time_ODE, Mask_ODE, CI=None,
                metabolite=False, full=False):

        stiff_ODE=False
        if self.config.method_solver in  ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
            from torchdiffeq import odeint_adjoint as odeint
            stiff_ODE=True
        elif self.config.solver == 'Adjoint':
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        latent_dim = self.config.latent_dim

        if CI is None:
            z_init = torch.zeros(b, latent_dim).to(self.device)
        else:
            z_init = torch.zeros(b, 3).to(self.device)
            z_init = torch.cat((z_init, CI), 1)
            assert z_init.shape[1] == latent_dim
        pred_z = z_init.unsqueeze(1).clone()

        for idx, (t0, t1) in enumerate(zip(Time_ODE[:-1], Time_ODE[1:])):

            # Dose is only added to the first compartment
            z_init[:, 0] = z_init[:, 0] + Doses_ODE[:, idx]
            time_interval = torch.Tensor([t0 - t0, t1 - t0]).to(self.device)

            if stiff_ODE:
                sol = odeint(self.ODE, z_init, time_interval, method='scipy_solver',
                            options={'solver': self.method}, rtol=self.rtol,
                            atol=self.atol).permute(1, 0, 2)
            else:
                sol = odeint(self.ODE, z_init, time_interval, rtol=self.rtol,
                             atol=self.atol, method=self.method).permute(1, 0, 2)
            z_init = sol[:, -1].clone()  # To avoid in-place operations
            pred_z = torch.cat((pred_z, sol[:, -1:, :]), 1)

        pred_z_out = []
        for id_ in range(b):
            pred_id = pred_z[id_][Mask_ODE[id_] == 1].view(-1, latent_dim)
            pred_z_out.append(pred_id)
        pred_z_out = pad_sequence(pred_z_out, batch_first=True)
        if pred_z_out.size(1) < self.config.max_l:
            np_, s_, d_ = pred_z_out.size()
            diff = self.config.max_l - s_
            zeros_ = torch.zeros(np_, diff, d_)
            pred_z_out = torch.cat((pred_z_out, zeros_), 1)
        
        pred_s_out = pred_z_out[:, :, 1]/self.ODE.V2_S
        pred_met_out = pred_z_out[:, :, 3]/self.ODE.V2_Met

        if metabolite and not full:
            return torch.stack((pred_s_out, pred_met_out), -1)
        elif metabolite and full:
            pred_full = pred_z.clone()
            pred_full = pred_full[:, :, 1]/self.ODE.V2_S
            return torch.stack((pred_s_out, pred_met_out), -1), pred_full
        elif full:
            pred_full = pred_z.clone()
            pred_full = pred_full[:, :, 1]/self.ODE.V2_S
            return pred_s_out, pred_full
        else:
            return pred_s_out

    def run(self):

        if self.config.solver == 'Adjoint':
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        global_steps = 0
        total_time = time.time()
        no_improvement = 0

        # Epoch_init begins in 1
        if self.config.epoch_init > 1:
            self.config.epoch_init += 1

        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):

            avg_loss = 0
            desc_bar = '[Iter: %d] Epoch: %d/%d' % (
                global_steps, epoch, self.config.num_epochs)

            progress_bar = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)

            epoch_time_init = time.time()

            # Training along dataset
            for iter, data in progress_bar:

                global_steps += 1

                Conc = data[0][0].to(self.device)        # Batch x Max_Num_Mea x 1
                COVL = data[1][0].to(self.device)        # Batch x Max_Num_Mea x DL
                MCOVL = data[2][0].to(self.device)       # Batch x Max_Num_Mea x DL
                COVS = data[3][0].to(self.device)        # Batch x Ds
                MCOVS = data[4][0].to(self.device)       # Batch x Ds
                Mask_Loss = data[5][0, :, :, 0].to(self.device)     # Batch x Time x 1
                Doses_ODE = data[6][0].to(self.device)   # Batch x Time_ODE x 1
                Mask_ODE = data[7][0].to(self.device)    # Batch x Time_ODE
                Time_ODE = data[8][0].to(self.device) # Time_ODE

                if global_steps == 1:
                    Time_DV = data[9][0, :, :, 0].to(self.device)  # Batch x Max_Num_Mea
                    self.data_fix = [
                        Conc.clone()[:10], COVL.clone()[:10], MCOVL.clone()[:10],
                        COVS.clone()[:10], MCOVS.clone()[:10], Mask_Loss.clone()[:10],
                        Doses_ODE.clone()[:10], Mask_ODE.clone()[:10],
                        Time_ODE.clone(), Time_DV[:10]]

                # First feature is always concentration
                # Last feature is always DiffTime
                _, eta_means, eta_log_vars = self.Enc(COVL, MCOVL, COVS, MCOVS)
                etas = self.Enc.reparameterize(eta_means, eta_log_vars)
                CI = None

                b = Conc.size(0)
                if self.config.type_ode == 'MathW_ODE':
                    COVS_ = self.Enc.Static_ImpLayer(COVS, MCOVS)
                    self.ODE.update_params(etas, COVS_[:, 2:3])
                else:
                    self.ODE.update_params(etas)
                pred_z = self.run_ODE(b, Doses_ODE, Time_ODE,
                                      Mask_ODE, CI=CI,
                                      metabolite=self.config.met_loss)

                if self.config.met_loss:
                    Meta = COVL[:, :, 1:2]
                    Conc = torch.cat((Conc, Meta), -1)
                    Mask_Loss = torch.stack((self.config.lambda_mse * Mask_Loss, self.config.lambda_mse_met*Mask_Loss), -1)
                else:
                    Conc = Conc[:, :, 0]
                    Mask_Loss = self.config.lambda_mse * Mask_Loss

                # ELBO LOSS = Likelihood + KL Div
                # Likelihood here is a MSE loss
                if self.config.w_loss:

                    norm_wl = pred_z if self.config.norm_w_loss == 'Pred' else Conc

                    if (norm_wl < 1e-3).sum() > 0:
                        # To avoid inf loss
                        mse = ((((pred_z - Conc)**2)/(norm_wl + 1e-3)) * Mask_Loss).sum(1).mean(0).sum(-1)
                    else:
                        mse = ((((pred_z - Conc)**2)/norm_wl) * Mask_Loss).sum(1).mean(0).sum(-1)
                else:
                    mse = ((pred_z - Conc)**2 * Mask_Loss).sum(1).mean(0).sum(-1)
                    if self.config.both_losses:
                        norm_wl = pred_z if self.config.norm_w_loss == 'Pred' else Conc
                        if (norm_wl < 1e-3).sum() > 0:
                            # To avoid inf loss
                            mse = 0.01*((((pred_z - Conc)**2)/(norm_wl + 1e-3)) * Mask_Loss).sum(1).mean(0).sum(-1)
                        else:
                            mse = 0.01*((((pred_z - Conc)**2)/Conc) * Mask_Loss).sum(1).mean(0).sum(-1)

                # KL Divergency
                p_mean = torch.ones_like(eta_means).to(self.device) * self.config.prior_mean
                p_std = torch.ones_like(eta_means).to(self.device) * self.config.prior_std

                prior_batch = dist.normal.Normal(p_mean, p_std)
                etas_batch = dist.normal.Normal(eta_means, torch.exp(0.5 * eta_log_vars))
                KL = (kl_divergence(etas_batch, prior_batch).sum(1).mean(0)) * self.config.lambda_kl
                loss = mse + KL

                if self.config.reg_L1:
                    L1 = sum(p.abs().sum() for p in self.Enc.parameters()) * self.config.lambda_reg_l1
                    loss = loss + L1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
                self.GPU_MEMORY_USED = self.get_gpu_memory_used()

                if (iter + 1) % self.config.print_freq == 0 or (iter + 1) == len(self.dataloader):
                    losses = OrderedDict()

                    losses['Used_GPU'] = colored(' {}MB'.format(self.GPU_MEMORY_USED), 'green')
                    losses['KL'] = KL.item()
                    losses['MSE'] = mse.item()
                    if self.config.reg_L1:
                        losses['L1'] = L1.item()
                    losses['Total Loss'] = loss.item()
                    losses['Average Loss'] = avg_loss / (iter + 1)
                    progress_bar.set_postfix(**losses)

            t_epoch = time.time() - epoch_time_init
            t_total = time.time() - total_time
            print_current_losses(epoch, global_steps, losses,
                t_epoch, t_total, self.config.save_path_losses,
                s_excel=True)

            avg_loss = avg_loss / (iter + 1)

            if (epoch) % self.config.save_freq == 0:
                self.save(epoch, avg_loss)

            if (epoch) % self.config.save_fig_freq == 0:
                self.save_fig_train(self.data_fix, epoch)

            if avg_loss < self.best_loss:
                self.save(epoch, avg_loss, best=True)
                self.best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement > self.config.patience:
                self.config.num_epochs = epoch
                print('Early Stoped in epoch ', epoch)
                break

        self.save(epoch, avg_loss)
        self.config.epoch_init = epoch

    @torch.no_grad()
    def validate(self):

        epoch = self.config.epoch_init
        desc_bar = '[Val] Epoch: %d' % (epoch)

        path_ = os.path.join(self.config.save_path_samples)
        self.save_ode_params(path_, epoch)

        progress_bar_val = tqdm(enumerate(self.dataloader_val),
                                unit_scale=True,
                                total=len(self.dataloader_val),
                                desc=desc_bar)

        fold = self.config.val_data_type
        name_ = 'Val_Imgs_%s'%(self.config.val_data_split)
        save_path = os.path.join(self.config.save_path_samples,
                                 name_, fold)
        os.makedirs(save_path, exist_ok=True)

        data_save, VI_save = {}, {}
        MAPE, SMAPE, RMSE = [], [], []

        if 'PostDist' in self.config.val_data_split:
            total_etas = []
            params_data_path = os.path.join(
                os.path.join(self.config.save_path_samples,
                             'Val_Imgs_Same_Patients'),
                'Params_Values_Ep%d.pth'%(self.config.epoch_init))

            params_data = torch.load(params_data_path)
            keys_groups = params_data.keys()
            for key_g in keys_groups:
                data_g = params_data[key_g]
                etas_g = data_g['Total_Etas']
                total_etas.append(etas_g)
                
            total_etas = torch.cat(total_etas)
            num_etas = total_etas.size(-1)
            total_etas = total_etas.view(-1, num_etas)
            means, stds = total_etas.mean(0), total_etas.std(0)
            self.Mean_Dists = dist.normal.Normal(means, stds)

        for iter, data in progress_bar_val:

            name = 'Num_DV_%d'%(data[0][0].shape[1])

            preds = self.get_pred_val(data, name, epoch, save_path, save_fig=True)

            (x, pred_x, pred_x_full, PRED, GOF, met, pred_met, GOF_MET,
                Mask_Loss, eta_means, eta_log_vars, total_etas, total_params,
                Weight, Doses, Tsld, Time_DV, Doses_ODE,
                Mask_ODE, Time_ODE, mape, smape, rmse) = preds

            VI_name = {}
            VI_name['Params'] = total_params.detach().cpu()
            VI_name['Pred_Means'] = eta_means.detach().cpu().float()
            VI_name['Pred_Log_Vars'] = eta_log_vars.detach().cpu().float()
            VI_name['Weight'] = Weight.detach().cpu().float()
            VI_name['Total_Etas'] = total_etas.detach().cpu().float()
            VI_save[name] = VI_name

            data_name = {}
            data_name['DVObs'] = x.detach().cpu().float()
            data_name['DVPred'] = pred_x.detach().cpu().float()
            data_name['PRED'] = PRED.detach().cpu().float()
            data_name['GOF'] = GOF.detach().cpu().float()
            data_name['MET'] = met.detach().cpu().float()
            data_name['METPred'] = pred_met.detach().cpu().float()
            data_name['GOF_MET'] = GOF_MET.detach().cpu().float()
            data_name['AMT'] = Doses.cpu()
            data_name['TSLD'] = Tsld.cpu()
            data_name['Time_DV'] = Time_DV.cpu()
            data_name['Mask_Loss'] = Mask_Loss.cpu()
            data_name['AMT_ODE'] = Doses_ODE.cpu()
            data_name['Mask_ODE'] = Mask_ODE.cpu()
            data_name['Time_ODE'] = Time_ODE.cpu()
            data_name['Total_ODE_Out'] = pred_x_full.cpu()
            data_name['epoch'] = epoch
            data_name['PID'] = data[10][0].cpu()
            data_save[name] = data_name

            MAPE.append(mape)
            SMAPE.append(smape)
            RMSE.append(rmse)

        # SAVING
        l_fold = len(fold)
        path_ = save_path[:-l_fold]
        path_ = os.path.join(path_, 'Params_Values_Ep%d.pth'%(epoch))
        torch.save(VI_save, path_)

        torch.save(data_save, 
            os.path.join(save_path, 'Results_Ep%d.pth'%(epoch)))

        ## METRICS
        # mape_avg = torch.mean(torch.cat(MAPE))
        # smape_avg = torch.mean(torch.cat(SMAPE))
        # rmse_avg = torch.mean(torch.cat(RMSE))

        # mape_avg = round(mape_avg.item(), 3)
        # smape_avg = round(smape_avg.item(), 3)
        # rmse_avg = round(rmse_avg.item(), 3)

        # save_metrics(epoch, mape_avg, smape_avg, rmse_avg, save_path)

        if 'Same_Patients' in self.config.val_data_split:
            path_ = os.path.join(self.config.save_path_samples)
            self.save_ode_params(path_, epoch)