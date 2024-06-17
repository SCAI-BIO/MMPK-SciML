#!/usr/bin/ipython
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *
import seaborn as sns
from glob import glob
import shutil
from parser import base_parser
import warnings
warnings.filterwarnings('ignore')

colors_ = ['#071D49', '#9CA4B6', '#004BFF', '#FF8307', 
           '#9200E6', '#1AC9A8', '#FFE664', '#D90F64',
           '#99B7FF', '#996633', '#FFFFFF']

def get_time_bin(time, bins):

    time_bin = []
    for i in range(len(time)):
        diff = abs(time[i] - bins)
        min_ = min(diff)

        bin_ = diff == min_
        bin_ = bins[bin_][0]
        time_bin.append(bin_)

    try:
        time_bin = torch.stack(time_bin)
    except:
        time_bin = np.array(time_bin)
    return time_bin

def individual_plots(data, ids, save_path):

    for id in ids:
        save_name = os.path.join(save_path, 'ID_%d.png'%(id))

        data_id = data[data.PTNO==id]
        plt.figure()
        sns.scatterplot(data=data_id, x='TIME', y='DV', color=colors_[0],
                        label='Observed data').set(xlabel=None)
        sns.lineplot(data=data_id, x='TIME', y='IPRED', color=colors_[4],
                     linestyle='dotted', estimator=np.median,
                     label='IPRED (Median + 95% PI)', err_style='band', errorbar=('pi', 95)).set(xlabel=None)
        sns.lineplot(data=data_id, x='TIME', y='PRED', color=colors_[6],
                     estimator=np.median, label='PRED').set(xlabel=None)
        plt.xlabel('Time in days')
        plt.ylabel('Drug concentration (μg/mL)')
        plt.legend(fontsize=8)
        plt.title('Patient Example %d'%(id))
        plt.savefig(save_name, dpi=400)
        plt.close()
        print('individual image is saved'%(id))


def pcVPC_DiffGroups(data, save_path, log_y=False, lb=0,
                     confidence=0.95, bins=[], xt='Days',
                     breaks=[], plot_data=False):

    alpha_2 = round((1 - confidence) / 2, 3)
    p_alpha_2 = 1 - alpha_2

    name = 'pcVPC.png'
    if log_y:
        name = 'LogY_%s'%(name)
    if plot_data:
        name = 'WithData_%s'%(name)
    save_name = os.path.join(save_path, name)

    df_obs = data[data.REP == 1].copy()

    # "left_join"
    df_obs['PREDmed'] = df_obs.groupby(['TIME_BIN'])['PRED'].transform('median')

    ObsStats = df_obs.copy()
    # Mutate
    ObsStats['DVcorr'] = lb + ((ObsStats.DV-lb) * (ObsStats.PREDmed-lb) / (ObsStats.PRED-lb))
    ObsStats['obsmed'] = ObsStats.groupby(['TIME_BIN'])['DVcorr'].transform('median')
    obsmed = ObsStats.groupby(['TIME_BIN'])['DVcorr'].median()
    obspmin = ObsStats.groupby(['TIME_BIN'])['DVcorr'].quantile(q=alpha_2)
    obspmax = ObsStats.groupby(['TIME_BIN'])['DVcorr'].quantile(q=p_alpha_2)
    tsld_u = df_obs.TIME_BIN.unique()
    tsld_u.sort()
    ObsStats = pd.DataFrame({
        "TIME_BIN":tsld_u, "obsmed":obsmed,
        "obspmin":obspmin, "obspmax":obspmax}).reset_index(drop=True)

    df_pred = data.copy()
    df_pred['PREDmed'] = df_pred.groupby(['TIME_BIN', 'REP'])['PRED'].transform('median')
    PredStats = df_pred.copy()
    PredStats['IPREDcorr'] = lb + ((PredStats.IPRED-lb) * (PredStats.PREDmed-lb) / (PredStats.PRED-lb))
    med = PredStats.groupby(['TIME_BIN', 'REP'])['IPREDcorr'].median()
    pmin = PredStats.groupby(['TIME_BIN', 'REP'])['IPREDcorr'].quantile(alpha_2)
    pmax = PredStats.groupby(['TIME_BIN', 'REP'])['IPREDcorr'].quantile(p_alpha_2)

    tsld_u = df_pred.TIME_BIN.unique()
    tsld_u.sort()
    tsld_df = pd.DataFrame({"TIME_BIN":tsld_u})
    rep_df = pd.DataFrame({"REP":df_pred.REP.unique()})

    index_ = pd.merge(tsld_df, rep_df, how='cross')
    metrics_= pd.DataFrame({"med":med, "pmin":pmin, "pmax":pmax}).reset_index(drop=True)
    PredStats = pd.concat((index_, metrics_), axis=1)

    medmed = PredStats.groupby(['TIME_BIN'])['med'].median()
    pminmed = PredStats.groupby(['TIME_BIN'])['med'].quantile(alpha_2)
    pmaxmed = PredStats.groupby(['TIME_BIN'])['med'].quantile(p_alpha_2)

    medpmin = PredStats.groupby(['TIME_BIN'])['pmin'].median()
    pminpmin = PredStats.groupby(['TIME_BIN'])['pmin'].quantile(alpha_2)
    pmaxpmin = PredStats.groupby(['TIME_BIN'])['pmin'].quantile(p_alpha_2)

    medpmax = PredStats.groupby(['TIME_BIN'])['pmax'].median()
    pminpmax = PredStats.groupby(['TIME_BIN'])['pmax'].quantile(alpha_2)
    pmaxpmax = PredStats.groupby(['TIME_BIN'])['pmax'].quantile(p_alpha_2)

    metrics_ = pd.concat((medmed, pminmed, pmaxmed, medpmin,  pminpmin, pmaxpmin, medpmax, pminpmax, pmaxpmax), axis=1).reset_index(drop=True)
    PredStats = pd.concat((tsld_df, metrics_), axis=1)
    PredStats.columns = ["TIME_BIN", "medmed", "pminmed", "pmaxmed", "medpmin",  "pminpmin", "pmaxpmin", "medpmax", "pminpmax", "pmaxpmax"]

    metrics_pcVPC = pd.merge(ObsStats, PredStats)

    log_y = True

    plot_ = ggplot(metrics_pcVPC) + \
        geom_errorbar(aes(x='TIME_BIN', ymin='obspmin', ymax='obspmax'), color="black", size = 0.5) + \
        geom_point(aes(x='TIME_BIN', y='obsmed'), color="black", size = 1) + \
        geom_line(aes(x='TIME_BIN', y='medmed'), color="purple") + \
        geom_ribbon(aes(x='TIME_BIN', ymin='pminmed', ymax='pmaxmed'), fill="purple",alpha=0.3) + \
        geom_line(aes(x='TIME_BIN', y='medpmin'), color="blue") + \
        geom_ribbon(aes(x='TIME_BIN',ymin='pminpmin', ymax='pmaxpmin'), fill="blue", alpha=0.3) + \
        geom_line(aes(x='TIME_BIN', y='medpmax'), color="blue") + \
        geom_ribbon(aes(x='TIME_BIN', ymin='pminpmax', ymax='pmaxpmax'), fill="blue", alpha=0.3) + \
        theme_bw() + scale_x_continuous(limits=[-1, 125], breaks=breaks) + \
        xlab("TIME (%s)"%(xt)) + ylab("Pred.-Corrected Drug Concentration (μg/mL)") + \
        labs(title='pcVPC')

    if log_y:
        plot_ += scale_y_log10(limits=[0.01, 1000], breaks=[0.01, 0.1, 1, 10, 100, 1000])

    if plot_data:
        plot_ += geom_point(data=df_obs, mapping=aes(x='TIME',y='DV'),alpha=0.15)
    plot_.save(save_name, height=5, width=7, dpi=400)


def Residual_plots(df, save_path, epoch=150):

    save_name = os.path.join(save_path, 'Residual_plots_Ep%d.png'%(epoch))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) # , sharex=True, sharey=True)

    Residual = df.DV - df.GOF_IPRED
    max_er, min_er = Residual.max(), Residual.min()
    t_max, tsld_max = df.TIME.max(), df.TSLD.max()
    axs[0].plot(df.TIME, Residual, '.', color=colors_[0], alpha=0.5)
    axs[1].plot(df.TSLD, Residual, '.', color=colors_[0], alpha=0.5)

    if max_er % 5 != 0:
        max_er = (max_er//5) * 5 + 5

    if min_er % 5 != 0:
        min_er = (min_er//5) * 5 - 5

    if t_max % 4 != 0:
        t_max = (t_max//4) * 4 + 4

    if tsld_max % 4 != 0:
        tsld_max = (tsld_max//4) * 4 + 4

    axs[0].axhline(y=0.0, color='r', linestyle='-')
    axs[1].axhline(y=0.0, color='r', linestyle='-')

    axs[0].set_xlabel('Time (day)', fontsize=8)
    axs[0].set_ylabel('Error (OBS - IPRED)', fontsize=8)
    axs[0].set_xlim(0, t_max)
    axs[0].set_ylim(min_er, max_er)
    axs[0].tick_params(labelsize=8)

    axs[1].set_xlabel('Time since last dose (day)', fontsize=8)
    axs[1].set_ylabel('Error (OBS - IPRED)', fontsize=8)
    axs[1].set_xlim(0, tsld_max)
    axs[1].set_ylim(min_er, max_er)
    axs[1].tick_params(labelsize=8)

    fig.suptitle('Residuals')

    plt.tight_layout()
    plt.savefig(save_name, dpi=400)
    plt.close()
    print('Residual image saved')


def GOF(df, save_path, name='Conc', log=False, epoch=150):

    name_ = 'GOF_%s_Ep%d.png'%(name, epoch)
    if log:
        name_ = 'Log_%s'%(name_)
    save_name = os.path.join(save_path, name_)

    real = df.DV
    pred = df.GOF_IPRED

    max_x = real.max()
    max_pred = pred.max()
    max_c = max_x if max_x >= max_pred else max_pred
    max_c += 0.2
    max_c = round(max_c, 2)

    min_x = real.min()
    min_pred = pred.min()
    min_c = min_x if min_x <= min_pred else min_pred
    min_c -= 0.2
    min_c = round(min_c, 2)

    df = pd.DataFrame({'Real': real, 'Pred': pred})
    plt.figure()
    # plt.plot(np.arange(min_c, max_c, 0.01), np.arange(min_c, max_c, 0.01), 'r')
    sns.scatterplot(df,  x='Real', y='Pred')

    if log:
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(np.arange(10e-4, 10e1, 0.01), np.arange(10e-4, 10e1, 0.01), 'r')
        plt.xlim((10e-4, 10e1))
        plt.ylim((10e-4, 10e1))
    else:
        plt.plot(np.arange(min_c, max_c, 0.01), np.arange(min_c, max_c, 0.01), 'r')
        plt.xlim((min_c, max_c))
        plt.ylim((min_c, max_c))

    plt.ylabel('Individual Predicted Drug Concentration (μg)')
    plt.xlabel('Observed Drug Concentration (μg)')
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()
    print('GOF image saved')

def save_metrics(save_path, epoch, df, metabolite=False):

    if metabolite:
        df = df[df.MET > 0]
        save_path = os.path.join(save_path, 'metrics_Met.csv')
        mape = 100 * abs((df.MET - df.GOF_MET)/df.MET).mean()
        smape = 100 * (abs(df.MET - df.GOF_MET) / (abs(df.MET) + abs(df.GOF_MET))).mean()
        rmse = np.sqrt(((df.MET - df.GOF_MET) ** 2).mean())
        mae = abs(df.MET - df.GOF_MET).mean()
        nmae = (abs(df.MET - df.GOF_MET).mean()) / (df.MET.mean())
    else:
        save_path = os.path.join(save_path, 'metrics.csv')
        mape = 100 * abs((df.DV - df.GOF_IPRED)/df.DV).mean()
        smape = 100 * (abs(df.DV - df.GOF_IPRED) / (abs(df.DV) + abs(df.GOF_IPRED))).mean()
        rmse = np.sqrt(((df.DV - df.GOF_IPRED) ** 2).mean())
        mae = abs(df.DV - df.GOF_IPRED).mean()
        nmae = (abs(df.DV - df.GOF_IPRED).mean()) / (df.DV.mean())

    mape = round(mape, 3)
    smape = round(smape, 3)
    rmse = round(rmse, 3)    
    mae = round(mae, 3)
    nmae = round(nmae, 3)
    metrics = [epoch, mape, smape, rmse, mae, nmae]

    import csv
    typ = 'a' if os.path.exists(save_path) else 'wt'
    header = ['Epoch', 'MAPE', 'SMAPE', 'RMSE', 'MAE', 'NMAE']

    with open(save_path, typ, encoding='UTF8') as file:

        writer = csv.writer(file)
        if typ == 'wt':
            writer.writerow(header)

        writer.writerow(metrics)

    file.close()

def main(config):
    print('Begin', config.val_data_type)

    # ===================================
    # ======== Load the data ============
    # ===================================

    # If there are more than 1 checkpoint, the model will do the plots
    # for all the saved checkpoints
    data = config.save_path
    new_path = data.split('/')[-1]
    new_path = data.replace(new_path, 'Individual_Variation')
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path, exist_ok=True)
    epoch = config.epoch
    data = torch.load(data)
    keys_groups = data.keys()

    PTNO, REP, TIME, TSLD, GID = [], [], [], [], []
    AMT, DV, IPRED, PRED, PRED_PTNO = [], [], [], [], []
    GOF_PTNO, MET, GOF_MET = [], [], []

    total_patients = 0
    group_names = {}
    for group_idx, key_g in enumerate(keys_groups):
        data_g = data[key_g]
        Obs = data_g['DVObs']
        IPred = data_g['DVPred']
        Doses = data_g['AMT'][..., 0]
        Time = data_g['Time_DV']
        Tsld = data_g['TSLD']
        GOF_pat = data_g['GOF']
        GOF_pat_met = data_g['GOF_MET']
        if config.MET_Covariates:
            Met = data_g['MET'][..., 0]
        else:
            Met = torch.zeros_like(GOF_pat_met) - 1

        Mean_Pat = data_g['PRED']
        PID = data_g['PID']
        PRED_ = Mean_Pat.mean(0)

        group_names[str(group_idx)] = key_g
        group_idx = torch.ones_like(PRED_) * group_idx

        patients_group = int(Obs.shape[0])
        for nr in range(config.nruns_ppd):
            ipred_rep = IPred[:, nr, :]
            obs_rep = Obs[:, nr, :]

            for ptno, ptid in enumerate(PID):
                obs_ptno = obs_rep[ptno]
                mask_ = obs_ptno > 0
                obs_ptno = obs_ptno[mask_ ==1]
                ipred_ptno = ipred_rep[ptno][mask_ ==1]
                tsld_ptno = Tsld[ptno][mask_ ==1]
                time_ptno = Time[ptno][mask_ ==1]
                PRED_ptno = Mean_Pat[ptno][mask_ ==1]
                GOF_ptno = GOF_pat[ptno][mask_ ==1]
                Met_ptno = Met[ptno][mask_ ==1]
                GOF_ptno_met = GOF_pat_met[ptno][mask_ ==1]

                l_ = obs_ptno.shape[0]
                dose_ptno = Doses[ptno][mask_ ==1]
                rep_ = torch.ones_like(dose_ptno) + nr
                ptno = torch.tensor([ptid]).repeat(l_) + total_patients
                Gid_ptno = group_idx[mask_ ==1]
                PRED_pred = PRED_[mask_ ==1]

                PTNO.append(ptno)
                REP.append(rep_)
                GID.append(Gid_ptno)
                TIME.append(time_ptno)
                TSLD.append(tsld_ptno)
                AMT.append(dose_ptno)
                DV.append(obs_ptno)
                PRED.append(PRED_pred)
                PRED_PTNO.append(PRED_ptno)
                GOF_PTNO.append(GOF_ptno)
                GOF_MET.append(GOF_ptno_met)
                MET.append(Met_ptno)
                IPRED.append(ipred_ptno)

        total_patients += patients_group
    
    PTNO = torch.cat(PTNO).numpy()
    REP = torch.cat(REP).numpy()
    GID = torch.cat(GID).numpy()
    TIME = torch.cat(TIME).numpy()
    TSLD = torch.cat(TSLD).numpy()
    AMT = torch.cat(AMT).numpy()
    DV = torch.cat(DV).numpy()
    PRED = torch.cat(PRED).numpy()
    IPRED = torch.cat(IPRED).numpy()
    PRED_PTNO = torch.cat(PRED_PTNO).numpy()
    GOF_PTNO = torch.cat(GOF_PTNO).numpy()
    MET = torch.cat(MET).numpy()
    GOF_MET = torch.cat(GOF_MET).numpy()
    df = pd.DataFrame({'PTNO':PTNO, 'REP':REP, 'TIME':TIME/24,
                       'TSLD':TSLD, 'AMT':AMT, 'DV':DV, 'PRED':PRED_PTNO,
                       'IPRED':IPRED, 'GOF_IPRED':GOF_PTNO, 'MET':MET,
                       'GOF_MET':GOF_MET})

    vpc_path = config.vpc_path
    path = os.path.join(vpc_path, 'VPCs')
    os.makedirs(path, exist_ok=True)

    df_rep1 = df[df.REP == 1].copy()
    save_metrics(save_path, epoch, df_rep1)
    save_metrics(save_path, epoch, df_rep1, metabolite=True)

    Residual_plots(df_rep1, path, epoch=epoch)
    GOF(df_rep1, path, log=True, epoch=epoch)
    GOF(df_rep1, path, epoch=epoch)

    # handling method for data below the limit of quantification
    loq = 2 * df[df.REP == 1].DV.min()
    # loq = 2 * 0.06
    df.loc[df.DV < loq, 'DV'] = loq/2
    df.loc[df.PRED < loq, 'PRED'] = loq/2
    df.loc[df.IPRED < loq, 'IPRED'] = loq/2

    if 'New' in config.val_data_split:
        bins = np.array([48, 312, 672, 1032, 1440, 2160])
        bins = np.array([2, 13, 28, 43, 62, 90])
    else:
        # day 1, 6, 14, 21, 28, 49, 64, 70, 90
        bins = np.array([24, 144, 336, 504, 672, 1176, 1536, 1680, 2160])
        bins = np.array([1, 6, 14, 21, 28, 49, 64, 70, 90])

    breaks = bins.copy()
    tsld_ = df.TIME.to_numpy()
    xt = 'days'
    time_bin = get_time_bin(tsld_, bins)
    df['TIME_BIN'] = time_bin
    
    path = os.path.join(vpc_path, 'VPCs_R')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'Data_Ep%d.csv'%(epoch))
    df.to_csv(path)

    path = os.path.join(vpc_path, 'VPCs')
    Math = True if 'Math' in config.type_ode else False
    pcVPC_DiffGroups(df, path, log_y=True, confidence=0.9,
                     bins=bins, xt=xt, breaks=breaks)
    pcVPC_DiffGroups(df, path, log_y=True, confidence=0.9,
                     bins=bins, xt=xt, breaks=breaks,
                     plot_data=True)

    path_indv = os.path.join(vpc_path, 'Extrapolation', 'Individual_Variation')
    ids = np.random.randint(df.PTNO.min(), df.PTNO.max() + 1, (5, ))
    individual_plots(df, ids, path_indv)

    print('Finish')


if __name__ == '__main__':

    config = base_parser()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.dataset,
                                    config.exp_name, 'Fold_%d'%(config.fold))
    config.save_path_samples = os.path.join(config.save_path, 'samples')

    fold = config.val_data_type
    name_ = 'Val_Imgs_%s'%(config.val_data_split)
    config.vpc_path = os.path.join(config.save_path_samples, name_)
    save_path = os.path.join(config.save_path_samples, name_, fold)

    # If there are more than 1 checkpoint, the model will do the plots
    # for all the saved checkpoints
    if config.from_best:
        opcs = glob(save_path + '/*.pth')
        epoch = 1
        for opc in opcs:
            epoch_opc = opc.split('/')[-1].split('Ep')[-1].split('.pth')[0]

            if int(epoch_opc) > epoch:
                epoch = int(epoch_opc)
        config.epoch = epoch
    assert config.epoch != 1
    config.save_path = os.path.join(
        save_path, 'Results_Ep%d.pth'%(config.epoch))
    main(config)
