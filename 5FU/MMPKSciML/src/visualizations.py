import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from plotnine import *
import torch


def scatter_params(real, pred, save_name, name='AUC'):

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
    plt.plot(np.arange(min_c, max_c, 0.01), np.arange(min_c, max_c, 0.01), 'r')
    sns.scatterplot(df,  x='Real', y='Pred')
    plt.xlabel('Real %s'%(name))
    plt.xlim((min_c, max_c))
    plt.ylim((min_c, max_c))
    plt.ylabel('Predicted %s'%(name))
    # plt.title('Real %s vs Predicted %s'%(name, name), fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def gof_plot(real, pred, save_name):

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
    plt.plot(np.arange(min_c, max_c, 0.01), np.arange(min_c, max_c, 0.01), 'r')
    sns.scatterplot(df,  x='Real', y='Pred')
    plt.ylabel('Individual Predicted Drug Concentration (μg/mL)')
    plt.xlabel('Observed Drug Concentration (μg/mL)')
    plt.xlim((min_c, max_c))
    plt.ylim((min_c, max_c))
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def plot_post_mean_var(mean, var_, save_name, nbins=10, name='CL'):

    bins_mean = np.histogram_bin_edges(mean, nbins)
    bins_var = np.histogram_bin_edges(var_, nbins)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    sns.histplot(mean, bins=bins_mean, kde=False, ax=axs[0]).set(xlabel=None)
    sns.histplot(var_, bins=bins_var, kde=False, ax=axs[1]).set(xlabel=None)

    axs[0].set_title('Posterior Mean %s distribution'%(name), fontsize=12)
    axs[1].set_title('Posterior Variance %s distribution (Log scale)'%(name), fontsize=12)
    axs[1].set_ylabel(' ', fontsize=12)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def plot_post_mean_and_real(real, mean, save_name, name='CL', nbins=10):

    bins_real = np.histogram_bin_edges(real, nbins)
    bins_mean = np.histogram_bin_edges(mean, nbins)
    plt.figure()
    sns.histplot(real, bins=bins_real, kde=False, label='Real %s distribution'%(name)).set(xlabel=None)
    sns.histplot(mean, bins=bins_mean, kde=False, label='Posterior Mean %s distribution'%(name)).set(xlabel=None)

    plt.title('Real %s and Posterior Mean %s Distributions (Log scale)'%(name, name), fontsize=14)
    plt.legend()
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()


def plot_post_mean(mean, save_name, name='V', nbins=10):

    bins_mean = np.histogram_bin_edges(mean, nbins)
    plt.figure()
    sns.histplot(mean, bins=bins_mean, kde=False).set(xlabel=None)

    plt.title('Posterior Mean %s Distribution (Log scale)'%(name), fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def post_Param_and_real(Param, save_name, nbins=10, name='V'):
    bins_pred = np.histogram_bin_edges(Param, nbins)

    plt.figure()
    sns.histplot(Param, bins=bins_pred, kde=False).set(xlabel=None)

    plt.title('Predicted %s Distribution'%(name), fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def etaDist(Param, save_name, nbins=10, xlim=[-2, 2],
            dname='Predicted', name='CL'):

    name_title = dname%(name)
    bins_pred = np.histogram_bin_edges(Param, nbins)
    plt.figure()
    sns.histplot(Param, bins=bins_pred, kde=True).set(xlabel=None)

    plt.xlim(xlim[0], xlim[1])
    plt.title(name_title, fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def pcVPC(data, epoch, save_path, log_y=False, lb=0,
          real_tsld=False, confidence=0.95, bins=[],
          plot_data=False):

    alpha_2 = round((1 - confidence) / 2, 3)
    p_alpha_2 = 1 - alpha_2

    name = 'pcVPC_Ep%d.png'%(epoch)
    if log_y:
        name = 'LogY_%s'%(name)
    if real_tsld:
        name = 'Real_TSLD_%s'%(name)
    if plot_data:
        name = 'WithData_%s'%(name)
    save_name = os.path.join(save_path, name)

    df_obs = data[data.REPI == 1].copy()
    # PredCorr = df_obs.groupby(['TSLD_BIN'])['PRED'].median()

    # "left_join"
    df_obs['PREDmed'] = df_obs.groupby(['TSLD_BIN'])['PRED'].transform('median')

    ObsStats = df_obs.copy()
    # Mutate
    ObsStats['DVcorr'] = lb + ((ObsStats.DVORIG-lb) * (ObsStats.PREDmed-lb) / (ObsStats.PRED-lb))
    ObsStats['obsmed'] = ObsStats.groupby(['TSLD_BIN'])['DVcorr'].transform('median')
    obsmed = ObsStats.groupby(['TSLD_BIN'])['DVcorr'].median()
    obspmin = ObsStats.groupby(['TSLD_BIN'])['DVcorr'].quantile(q=alpha_2)
    obspmax = ObsStats.groupby(['TSLD_BIN'])['DVcorr'].quantile(q=p_alpha_2)
    tsld_u = df_obs.TSLD_BIN.unique()
    tsld_u.sort()
    ObsStats = pd.DataFrame({
        "TSLD_BIN":tsld_u, "obsmed":obsmed,
        "obspmin":obspmin, "obspmax":obspmax}).reset_index(drop=True)

    df_pred = data.copy()
    df_pred['PREDmed'] = df_pred.groupby(['TSLD_BIN', 'REPI'])['PRED'].transform('median')
    PredStats = df_pred.copy()
    PredStats['IPREDcorr'] = lb + ((PredStats.IPRED-lb) * (PredStats.PREDmed-lb) / (PredStats.PRED-lb))
    med = PredStats.groupby(['TSLD_BIN', 'REPI'])['IPREDcorr'].median()
    pmin = PredStats.groupby(['TSLD_BIN', 'REPI'])['IPREDcorr'].quantile(alpha_2)
    pmax = PredStats.groupby(['TSLD_BIN', 'REPI'])['IPREDcorr'].quantile(p_alpha_2)

    tsld_u = df_pred.TSLD_BIN.unique()
    tsld_u.sort()
    tsld_df = pd.DataFrame({"TSLD_BIN":tsld_u})
    rep_df = pd.DataFrame({"REPI":df_pred.REPI.unique()})

    index_ = pd.merge(tsld_df, rep_df, how='cross')
    metrics_= pd.DataFrame({"med":med, "pmin":pmin, "pmax":pmax}).reset_index(drop=True)
    PredStats = pd.concat((index_, metrics_), axis=1)

    medmed = PredStats.groupby(['TSLD_BIN'])['med'].median()
    pminmed = PredStats.groupby(['TSLD_BIN'])['med'].quantile(alpha_2)
    pmaxmed = PredStats.groupby(['TSLD_BIN'])['med'].quantile(p_alpha_2)

    medpmin = PredStats.groupby(['TSLD_BIN'])['pmin'].median()
    pminpmin = PredStats.groupby(['TSLD_BIN'])['pmin'].quantile(alpha_2)
    pmaxpmin = PredStats.groupby(['TSLD_BIN'])['pmin'].quantile(p_alpha_2)

    medpmax = PredStats.groupby(['TSLD_BIN'])['pmax'].median()
    pminpmax = PredStats.groupby(['TSLD_BIN'])['pmax'].quantile(alpha_2)
    pmaxpmax = PredStats.groupby(['TSLD_BIN'])['pmax'].quantile(p_alpha_2)

    metrics_ = pd.concat((medmed, pminmed, pmaxmed, medpmin,  pminpmin, pmaxpmin, medpmax, pminpmax, pmaxpmax), axis=1).reset_index(drop=True)
    PredStats = pd.concat((tsld_df, metrics_), axis=1)
    PredStats.columns = ["TSLD_BIN", "medmed", "pminmed", "pmaxmed", "medpmin",  "pminpmin", "pmaxpmin", "medpmax", "pminpmax", "pmaxpmax"]

    metrics_pcVPC = pd.merge(ObsStats, PredStats)
    log_y = True
    if len(metrics_pcVPC) > 1:
        # geom_point(aes(x='TSLD_BIN', y='obspmin'), color="black", size = 1) + \
        # geom_point(aes(x='TSLD_BIN', y='obspmax'), color="black", size = 1) + \
        plot_ = ggplot(metrics_pcVPC) + \
            geom_errorbar(aes(x='TSLD_BIN', ymin='obspmin', ymax='obspmax'), color="black", size = 0.5) + \
            geom_point(aes(x='TSLD_BIN', y='obsmed'), color="black", size = 1) + \
            geom_line(aes(x='TSLD_BIN', y='medmed'), color="purple") + \
            geom_ribbon(aes(x='TSLD_BIN', ymin='pminmed', ymax='pmaxmed'), fill="purple",alpha=0.3) + \
            geom_line(aes(x='TSLD_BIN', y='medpmin'), color="blue") + \
            geom_ribbon(aes(x='TSLD_BIN',ymin='pminpmin', ymax='pmaxpmin'), fill="blue", alpha=0.3) + \
            geom_line(aes(x='TSLD_BIN', y='medpmax'), color="blue") + \
            geom_ribbon(aes(x='TSLD_BIN', ymin='pminpmax', ymax='pmaxpmax'), fill="blue", alpha=0.3) + \
            theme_bw() + scale_x_continuous(breaks=bins) + \
            xlab("Time since last dose (hours)") + ylab("Pred.-Corrected Drug Concentration (μg/mL)") + \
            labs(title='pcVPC')
    else:
        plot_ = ggplot(metrics_pcVPC) + \
            geom_errorbar(aes(x='TSLD_BIN', ymin='obspmin', ymax='obspmax'), color="black", size = 0.5) + \
            geom_point(aes(x='TSLD_BIN', y='obsmed'), color="black", size = 1) + \
            geom_point(aes(x='TSLD_BIN', y='medmed'), color="purple", size = 1) + \
            geom_errorbar(aes(x='TSLD_BIN', ymin='pminmed', ymax='pmaxmed'), color="purple", alpha=0.3) + \
            geom_errorbar(aes(x='TSLD_BIN', ymin='medpmin', ymax='medpmax'), color="blue", size = 0.5) + \
            geom_errorbar(aes(x='TSLD_BIN', ymin='pminpmin', ymax='pmaxpmin'), color="blue", alpha=0.3) + \
            geom_errorbar(aes(x='TSLD_BIN', ymin='pminpmax', ymax='pmaxpmax'), color="blue", alpha=0.3) + \
            theme_bw() + scale_x_continuous(breaks=bins) + \
            xlab("Time since last dose (hours)") + ylab("Pred.-Corrected Drug Concentration (μg/mL)") + \
            labs(title='pcVPC')
        
    if log_y:
        plot_ += scale_y_log10()
    if plot_data:
        plot_ += geom_point(data=df_obs, mapping=aes(x='TSLD_REAL',y='DVORIG'), alpha=0.3)
    plot_.save(save_name, height=5, width=7, dpi=400)


def individual_plots(data, ids, save_path, md=False):

    if md:
        aux_n = ' (Median + 95% PI)'
    else:
        aux_n = ''

    for id in ids:
        save_name = os.path.join(save_path, 'ID_%d.png'%(id))

        data_id = data[data.PTNO==id]
        max_t = data_id.TIME.values[-1] // 2
        max_t = 2 + (max_t * 2)

        max_x = data_id.DV.max()
        max_ipred = data_id.IPRED.max()
        max_pred = data_id.PRED.max()
        max_c = max_x if max_x >= max_ipred else max_ipred
        max_c = max_c if max_c >= max_pred else max_pred
        max_c = max_c // 1
        max_c = 1 + (max_c * 1)

        plt.figure()
        sns.scatterplot(data=data_id[data_id.DV > 0], x='TIME', y='DV',
                        label='Observed data', color='black').set(xlabel=None)
        sns.lineplot(data=data_id, x='TIME', y='IPRED',
                     linestyle='dotted', estimator=np.median,
                     label='IPRED'+ aux_n, err_style='band',
                     errorbar=('pi', 95)).set(xlabel=None)
        sns.lineplot(data=data_id, x='TIME', y='PRED',
                     estimator=np.median, label='PRED'+ aux_n).set(xlabel=None)
        plt.xlabel('Time (hours)')
        plt.ylabel('Drug concentration (ng/mL)')
        plt.xlim((0, max_t))
        plt.ylim((0, max_c))
        plt.legend(fontsize=8)
        plt.title('Patient Example %d'%(id))
        plt.savefig(save_name, dpi=400)
        plt.close()
        print('individual image is saved'%(id))
