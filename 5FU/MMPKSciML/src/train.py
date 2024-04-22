import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch
from torch.distributions import kl_divergence
import torch.nn.functional as F
from collections import OrderedDict
from termcolor import colored
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
    def get_tsld_bin(self, tsld, bins):

        tsld_bin = []
        for i in range(len(tsld)):
            diff = abs(tsld[i] - bins)
            min_ = min(diff)

            bin_ = diff == min_
            bin_ = bins[bin_][0]
            tsld_bin.append(bin_)

        try:
            tsld_bin = torch.stack(tsld_bin)
        except:
            tsld_bin = np.array(tsld_bin)
        return tsld_bin

    @torch.no_grad()
    def save_fig(self, data, epoch, val=False):

        if val:
            name_ = 'Val_Imgs_%s'%(self.config.val_data_split)
            save_path = os.path.join(self.config.save_path_samples,name_)
            T_Fig = self.Tval
            nruns_ppd = self.config.nruns_ppd_pp
        else:
            save_path = os.path.join(self.config.save_path_samples,'Train_Imgs')
            T_Fig = self.T1
            nruns_ppd = 1

        os.makedirs(save_path, exist_ok=True)

        Inps = data[0]      # Batch x Dims
        Conc = data[1]      # Batch x 1
        Doses = data[2]     # Batch x 1
        AUC = data[3]       # Batch x 1
        CL = data[4]        # Batch x 1
        if val:
            real_TSLD = data[5]

        b = Conc.size(0)
        total_real = list()
        total_pred = list()
        total_CL = list()
        total_AUC = list()
        total_V = list()
        total_etas = list()

        # nruns samples to approximate the mean of the posterior distribution
        # per patient
        _, eta_means, eta_log_vars = self.Enc(Inps)
        patient_dist = dist.normal.Normal(eta_means, torch.exp(0.5*eta_log_vars))
        if 'Prior' in self.config.val_data_split:
            p_mean = torch.ones_like(eta_means).to(self.device) * self.config.prior_mean
            p_std = torch.ones_like(eta_means).to(self.device) * self.config.prior_std
            patient_dist = dist.normal.Normal(p_mean, p_std)
        elif 'PostDist' in self.config.val_data_split:
            means = [], [], [], []
            results_SP = os.path.join(
                os.path.join(self.config.save_path_samples,
                             'Val_Imgs_Same_Patients'),
                'Results_Ep%d.pth'%(self.config.epoch_init))
        
            TP_data = torch.load(results_SP)
            etas_TP = TP_data['Total_Etas']
            num_etas = etas_TP.size(-1)
            etas_TP = etas_TP.view(-1, num_etas)
            means, std = etas_TP.mean(0), etas_TP.std(0)
            patient_dist = dist.normal.Normal(means, std)

        for nrun in range(nruns_ppd):

            if 'PostDist' in self.config.val_data_split:
                params = []
                for _ in range(b):
                    params.append(patient_dist.sample())
                params = torch.stack(params, 0)
            else:
                params = patient_dist.sample()
            total_etas.append(params.unsqueeze(1))
            self.ODE.update_params(params)

            # ODE
            b = Conc.size(0)
            pred_z = self.run_ODE(b, Doses, T_Fig)
            pred_AUC = Doses / self.ODE.CL

            # total real and total pred have the complete time series
            # according to the val time horizon
            total_real.append(Conc)
            total_pred.append(pred_z.unsqueeze(-1))
            total_CL.append(self.ODE.CL)
            total_AUC.append(pred_AUC)
            if self.config.type_ode == 'PPCL_PopV':
                V_ = torch.tensor(self.ODE.CL.shape[0]*[self.ODE.V]).unsqueeze(-1)
                total_V.append(V_)
            else:
                total_V.append(self.ODE.V)

        Conc = torch.cat(total_real, 1).detach().cpu().float()
        pred_Conc = torch.cat(total_pred, 1).detach().cpu().float()
        total_CL = torch.cat(total_CL, 1).detach().cpu().float()
        total_AUC = torch.cat(total_AUC, 1).detach().cpu().float()
        total_V = torch.cat(total_V, 1).detach().cpu().float()
        total_etas = torch.cat(total_etas, 1)

        AUC = AUC[:, 0].cpu().float()
        CL = CL[:, 0].cpu().float()

        if val:

            path_ = os.path.join(save_path, 'Etas')
            os.makedirs(path_, exist_ok=True)

            if AUC.shape[0] < 100:
                nbins = 10
            else:
                nbins = 20

            ename = '%s Eta'
            eta_CL = torch.log(total_CL.view(-1) / torch.exp(self.ODE.CL_pop)).numpy()
            save_name = os.path.join(path_, 'Eta_CL.png')
            etaDist(eta_CL, save_name, nbins=nbins, xlim=[-4, 4],
                    dname=ename, name='CL')
            if 'CLV' in self.config.type_ode:
                eta_V = torch.log(total_V.view(-1) / torch.exp(self.ODE.V_pop)).numpy()
                save_name = os.path.join(path_, 'Eta_V.png')
                etaDist(eta_V, save_name, nbins=nbins, xlim=[-3, 3],
                    dname=ename, name='V')
            else:
                eta_V = None

            # The metrics as well as the GOF plots need to be done with the
            # patient specific mean
            # Calculating the Trajectory with the Eta mean
            self.ODE.update_params(eta_means)
            pred_CL = self.ODE.CL
            pred_CL = pred_CL[:, 0].numpy()
            GOF_SEQ = self.run_ODE(b, Doses, T_Fig)
            pred_AUC = (Doses[:, 0].numpy() / pred_CL)
            AUC, CL = AUC.numpy(), CL.numpy()

            self.save_ode_params(save_path, epoch)
            save_metrics(CL, pred_CL, save_path, epoch, name_='CL')           
            self.save_dose_adjustments(AUC, pred_AUC, save_path)

            save_name = os.path.join(save_path, 'GOF_AUC_Ep%d.png'%(epoch))
            scatter_params(AUC, pred_AUC, save_name, name='AUC')

            save_name = os.path.join(save_path, 'GOF_CL_Ep%d.png'%(epoch))
            scatter_params(CL, pred_CL, save_name, name='CL')

            if 'CLV' in self.config.type_ode:
                V = self.ODE.V.numpy()   
                save_name = os.path.join(save_path, 'V_Distribution_Ep%d.png'%(epoch))
                post_Param_and_real(V, save_name, name='Volume')

            # Calculating the PRED           
            # The Etas need to be 0
            PRED_params = torch.zeros(b, self.config.dim_params).to(self.device)
            self.ODE.update_params(PRED_params)
            PRED_SEQ = self.run_ODE(b, Doses, T_Fig)
            PRED_ = PRED_SEQ.mean().detach().cpu().float()

            data_save = {}
            data_save['AUC'] = AUC
            data_save['Pred_AUC'] = total_AUC
            data_save['Conc'] = Conc
            data_save['Pred_Conc'] = pred_Conc
            data_save['PRED'] = PRED_SEQ
            data_save['GOF'] = GOF_SEQ
            data_save['CL'] = CL
            data_save['Pred_CL'] = total_CL
            data_save['Pred_V'] = total_V
            data_save['Eta_CL'] = eta_CL
            data_save['Eta_V'] = eta_V
            data_save['Doses'] = Doses.cpu()
            data_save['epoch'] = epoch
            data_save['Means'] = eta_means.detach().cpu().float()
            data_save['Log_Vars'] = eta_log_vars.detach().cpu().float()
            data_save['Total_Etas'] = total_etas.detach().cpu().float()

            torch.save(data_save, 
                os.path.join(save_path, 'Results_Ep%d.pth'%(epoch)))
            
            T_Fig = T_Fig.detach().cpu().float()
            PTNO, REP, TIME, TSLD, TSLD_REAL = [], [], [], [], []
            AMT, DV, IPRED, PRED, PRED_PTNO = [], [], [], [], []
            GOF_PTNO = []
            pat_num = Conc.shape[0]

            for nrun in range(nruns_ppd):
                obs_rep = Conc[:, nrun]
                ipred_rep = pred_Conc[:, nrun]

                for ptno in range(pat_num):
                    obs_ptno = obs_rep[ptno]
                    ipred_ptno = ipred_rep[ptno, 0]
                    dose_ptno = Doses[ptno, 0]
                    PRED_ptno = PRED_SEQ[ptno, 0]
                    GOF_ptno = GOF_SEQ[ptno, 0]

                    PTNO.append(torch.tensor(ptno))
                    REP.append(torch.tensor(nrun + 1))
                    AMT.append(dose_ptno)
                    DV.append(obs_ptno)
                    IPRED.append(ipred_ptno)
                    PRED.append(PRED_)
                    PRED_PTNO.append(PRED_ptno)
                    GOF_PTNO.append(GOF_ptno)
                    TIME.append(T_Fig[-1]*24)
                    TSLD.append(T_Fig[-1]*24)
                    TSLD_REAL.append(real_TSLD[ptno, 0])

            PTNO = torch.stack(PTNO).numpy()
            REP = torch.stack(REP).numpy()
            TIME = torch.stack(TIME).numpy()
            TSLD = torch.stack(TSLD).numpy()
            TSLD_REAL = torch.stack(TSLD_REAL).numpy()
            AMT = torch.stack(AMT).numpy()
            DV = torch.stack(DV).numpy()
            PRED = torch.stack(PRED).numpy()
            IPRED = torch.stack(IPRED).numpy()
            PRED_PTNO = torch.stack(PRED_PTNO).numpy()
            GOF_PTNO = torch.stack(GOF_PTNO).numpy()
            
            data = pd.DataFrame({'PTNO':PTNO, 'REP': REP, 'TSLD_ODE': TSLD,
                                 'TSLD_REAL': TSLD_REAL, 'TIME':TIME, 'AMT':AMT,
                                 'DV': DV, 'PRED': PRED_PTNO, 'IPRED': IPRED,
                                 'GOF_IPRED':GOF_PTNO})

            data_rep1 = data[data.REP == 1]
            save_metrics(data_rep1.DV, data.GOF_IPRED, save_path, epoch, name_='Conc')

            path_ = os.path.join(save_path, 'pcVPC')
            os.makedirs(path_, exist_ok=True)

            data_ode_tsld = data.copy()
            data_ode_tsld = data_ode_tsld.rename(columns={'TSLD_ODE': 'TSLD_BIN'})
            pcVPC(data_ode_tsld, epoch, path_, log_y=True, lb=0, confidence=0.9)

            data_real_tsld = data.copy()

            bins = np.array((17, 18, 18.5, 20))
            if 'New' in self.config.val_data_split:
                if self.config.fold == 2:
                    bins = np.array((17.5, 18, 19, 21))
                elif self.config.fold == 3:
                    bins = np.array((18, 18.5, 19, 20))
                elif self.config.fold == 4:
                    bins = np.array((17.5, 18, 18.5, 20))
                elif self.config.fold == 5:
                    bins = np.array((17, 18, 18.5, 19.5))
                elif self.config.fold == 6:
                    bins = np.array((18,19,21))
                elif self.config.fold == 7:
                    bins = np.array((18, 19, 21))
                elif self.config.fold == 10:
                    bins = np.array((17.5, 18, 18.5, 19.2))
            
            tsld_ = data_real_tsld.TSLD_REAL.to_numpy()
            tsld_bin = self.get_tsld_bin(tsld_, bins)
            data_real_tsld['TSLD_BIN'] = tsld_bin

            pcVPC(data_real_tsld, epoch, path_, log_y=True,
                  real_tsld=True, lb=0, confidence=0.9,
                  bins=bins)
            pcVPC(data_real_tsld, epoch, path_, log_y=True,
                  real_tsld=True, lb=0, confidence=0.9,
                  bins=bins, plot_data=True)

            save_name = os.path.join(save_path, 'GOF_Conc_Ep%d.png'%(epoch))
            gof_plot(data_rep1.DV, data.GOF_IPRED, save_name)
        else:

            AUC, CL = AUC.numpy(), CL.numpy()
            Conc = Conc[:, 0].numpy()
            pred_z = pred_Conc[:, 0, 0].numpy()
            pred_AUC = total_AUC[:, 0].numpy()
            pred_CL = total_CL[:, 0].numpy()
            V = total_V[:, 0].numpy()   

            save_name = os.path.join(save_path, 'GOF_AUC_Ep%d.png'%(epoch))
            scatter_params(AUC, pred_AUC, save_name, name='AUC')

            save_name = os.path.join(save_path, 'GOF_CL_Ep%d.png'%(epoch))
            scatter_params(CL, pred_CL, save_name, name='CL')

            save_name = os.path.join(save_path, 'GOF_Conc_Ep%d.png'%(epoch))
            gof_plot(Conc, pred_z, save_name)

            if 'CLV' in self.config.type_ode:
                save_name = os.path.join(save_path, 'V_Distribution_Ep%d.png'%(epoch))
                post_Param_and_real(V, save_name, name='Volume')


    @torch.no_grad()
    def save_dose_adjustments(self, rec_AUC, pred_AUC, save_path):

        ra_rec_AUC = len(rec_AUC[rec_AUC < 20]) # Raise dose
        re_rec_AUC = len(rec_AUC[rec_AUC > 30]) # Reduce dose
        k_rec_AUC = len(rec_AUC) - ra_rec_AUC - re_rec_AUC # keep dose

        ra_pred_AUC = len(pred_AUC[pred_AUC < 20]) # Raise dose
        re_pred_AUC = len(pred_AUC[pred_AUC > 30]) # Reduce dose
        k_pred_AUC = len(pred_AUC) - ra_pred_AUC - re_pred_AUC

        data = [re_rec_AUC, k_rec_AUC, ra_rec_AUC, re_pred_AUC, k_pred_AUC, ra_pred_AUC]
        save_path = os.path.join(save_path, 'Dose_Adjustment.csv')


        import csv
        typ = 'a' if os.path.exists(save_path) else 'wt'
        header = ['Rec_Reduce', 'Rec_Keep', 'Rec_Raise', 'Pred_Reduce', 'Pred_Keep', 'Pred_Raise']
        with open(save_path, typ, encoding='UTF8') as file:

            writer = csv.writer(file)
            if typ == 'wt':
                writer.writerow(header)
            writer.writerow(data)

    @torch.no_grad()
    def save_ode_params(self, save_path, epoch):

        save_path = os.path.join(save_path, 'ODE_Params.csv')
        CL = round(self.ODE.CL_pop.item(), 3)
        V = round(self.ODE.V_pop.item(), 3)
        data = [epoch, V, CL]

        print('V: ', V)
        print('CL: ', CL)

        import csv
        typ = 'a' if os.path.exists(save_path) else 'wt'
        header = ['Epoch', 'V', 'CL']
        with open(save_path, typ, encoding='UTF8') as file:

            writer = csv.writer(file)
            if typ == 'wt':
                writer.writerow(header)
            writer.writerow(data)

    def run_ODE(self, b, Doses, Time_ODE):

        if self.config.solver == 'Adjoint':
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # ODE
        z_init = torch.zeros(b, self.config.num_IC).to(self.device)
        pred_z = z_init.unsqueeze(1).clone()

        for idx, (t0, t1) in enumerate(zip(Time_ODE[:-1], Time_ODE[1:])):

            self.ODE.update_dose(Doses[:, 0])
            time_interval = torch.Tensor([t0 - t0, t1 - t0]).to(self.device)
            sol = odeint(self.ODE, z_init, time_interval, rtol=self.rtol,
                        atol=self.atol, method=self.method).permute(1, 0, 2)
            z_init = sol[:, -1].clone()  # To avoid in-place operations
            pred_z = torch.cat((pred_z, sol[:, -1:, :]), 1)

        pred_z = pred_z[:, -1:, -1]/self.ODE.V
        return pred_z

    def run(self):

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
                Inps = data[0].to(self.device)        # Batch x Dims
                Conc = data[1].to(self.device)        # Batch x 1
                Doses = data[2].to(self.device)       # Batch x 1

                if global_steps == 1:
                    AUC = data[3].to(self.device)        # Batch x 1
                    CL = data[4].to(self.device)         # Batch x 1

                    self.data_fix = [Inps.clone(), Conc.clone(), Doses.clone(),
                                     AUC.clone(), CL.clone()]

                # params = self.Enc(Inps)
                _, eta_means, eta_log_vars = self.Enc(Inps)
                params = self.Enc.reparameterize(eta_means, eta_log_vars)
                self.ODE.update_params(params)

                # ODE
                b = Conc.size(0)
                pred_z = self.run_ODE(b, Doses, self.T1)

                if self.config.type_loss == 'WMSE':
                    mse = self.config.lambda_mse * (((pred_z - Conc)**2)/Conc).sum(-1).mean()
                elif self.config.type_loss == 'L1':
                    mse = self.config.lambda_mse * (pred_z - Conc).abs().sum(-1).mean()
                else:
                    mse = self.config.lambda_mse * ((pred_z - Conc)**2).sum(-1).mean()

                # KL Divergency
                p_mean = torch.ones_like(eta_means).to(self.device) * self.config.prior_mean
                p_std = torch.ones_like(eta_means).to(self.device) * self.config.prior_std
                prior_batch = dist.normal.Normal(p_mean, p_std)
                dists_batch = dist.normal.Normal(eta_means, torch.exp(0.5 * eta_log_vars))
                KL = kl_divergence(dists_batch, prior_batch).sum(1).mean(0) * self.config.lambda_kl
                loss = mse + KL

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
                self.save_fig(self.data_fix, epoch)
            
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

        progress_bar_val = tqdm(enumerate(self.dataloader_val),
                                unit_scale=True,
                                total=len(self.dataloader_val),
                                desc=desc_bar)

        for iter, data in progress_bar_val:

            Inps = data[0].to(self.device)        # Batch x Dims
            Conc = data[1].to(self.device)        # Batch x 1
            Doses = data[2].to(self.device)       # Batch x 1
            AUC = data[3].to(self.device)         # Batch x 1
            CL = data[4].to(self.device)          # Batch x 1
            real_TSLD = data[5].to(self.device)          # Batch x 1

            data = [Inps, Conc, Doses, AUC, CL, real_TSLD]
            self.save_fig(data, epoch, val=True)

