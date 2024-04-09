import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import torch

colors_ = ['#071D49', '#9CA4B6', '#004BFF', '#FF8307', 
           '#9200E6', '#1AC9A8', '#FFE664', '#D90F64',
           '#99B7FF', '#996633', '#FFFFFF']

@torch.no_grad()
def pearson_corr(X, pred_x):
    total_pearson_corr = 0
    for i in range(X.size(0)):
        total_pearson_corr += pearsonr(X[i, :, 0].numpy(), pred_x[i, :, 0].numpy())[0]

    total_pearson_corr /= X.size(0)
    print('Pearson Corr', total_pearson_corr)


@torch.no_grad()
def patient_traj(X, pred_x, Doses, T_Fig,
                 xlim, epoch, save_path, n_figs=300,
                 train_time_max=None,
                 time_effect=None):

    Doses = Doses[:, 0, 0]

    # Information for axis
    max_x = X.max()
    max_pred = pred_x.max()
    max_c = max_x if max_x >= max_pred else max_pred

    if max_c % 5 != 0:
        max_c = (max_c//5) * 5 + 5
    
    t_org = T_Fig.clone().cpu().numpy()

    pred_batch = pred_x.shape[0]
    num_imgs = n_figs if pred_batch > n_figs else pred_batch

    save_path = os.path.join(save_path, 'Individual', str(epoch))
    os.makedirs(save_path, exist_ok=True)

    pat_num = [0, 1]
    # for i in range(num_imgs):
    for i in pat_num:
        save_name = os.path.join(save_path, 'ID_%d.png'%(i+1))
        x_id = X[i, :, 0]
        pred_id = pred_x[i, :, 0]
        real_df = pd.DataFrame(
            {'ID': i, 'Time': t_org, 'Real': x_id, 'Pred': pred_id})

        plt.figure()
        sns.lineplot(data=real_df, x='Time', y='Real', color=colors_[0], label='Observed Data',
                     estimator=np.median, err_style='band', errorbar=('pi', 95))
        sns.lineplot(data=real_df, x='Time', y='Pred', color=colors_[0], linestyle='dotted',
                     estimator=np.median, err_style='band', errorbar=('pi', 95),
                     label='Predicted Data')

        if train_time_max is not None:
            plt.axvline(x = train_time_max, color = colors_[8], linestyle='--', label='T train')
        if time_effect is not None:
            plt.axvline(x = time_effect, color = colors_[9], linestyle='--', label='T effect')
        plt.xlabel('Time in days')
        plt.ylabel('Drug concentration (µg/mL)')
        plt.xlim(0, xlim)
        plt.ylim(0, max_c)
        plt.legend(fontsize=8)
        plt.title('Patient ID %d Dose %d mg'%(i+1, Doses[i]))
        plt.savefig(save_name, dpi=400)
        plt.close()
        print('individual image %d is saved'%(i+1))


@torch.no_grad()
def GOF_plots(X, pred_x, T_data, TSLD, epoch, save_name, val=False):

    # Here X is the original data from the CSV which
    # also contains the random effect part
    t_data = T_data.clone().cpu()
    t_max = t_data.max()
    tsld = TSLD.clone().cpu()
    tsld_max = tsld.max()

    error = X - pred_x

    # Information for axis
    max_x = X.max()
    max_pred = pred_x.max()
    max_c = max_x if max_x >= max_pred else max_pred
    max_er, min_er = error.max(), error.min()

    if max_c % 5 != 0:
        max_c = (max_c//5) * 5 + 5

    if max_er % 5 != 0:
        max_er = (max_er//5) * 5 + 5

    if min_er % 5 != 0:
        min_er = (min_er//5) * 5 - 5
        
    if t_max % 4 != 0:
        t_max = (t_max//4) * 4 + 4

    if tsld_max % 4 != 0:
        tsld_max = (tsld_max//4) * 4 + 4

    fig, axs = plt.subplots(3, 2, figsize=(8, 7)) # , sharex=True, sharey=True)

    for i in range(len(X)):
        axs[0, 0].plot(pred_x[i, :], X[i, :], '.', color=colors_[0])
        axs[0, 1].plot(pred_x[i, :], X[i, :], '.', color=colors_[0])
        axs[1, 0].plot(pred_x[i, :], error[i, :], '.', color=colors_[0])
        axs[1, 1].plot(pred_x[i, :], error[i, :], '.', color=colors_[0])
        axs[2, 0].plot(t_data[i, :], error[i, :], '.', color=colors_[0])
        axs[2, 1].plot(tsld[i, :], error[i, :], '.', color=colors_[0])

    axs[0, 0].plot(torch.range(0, max_c), torch.range(0, max_c), colors_[7])
    axs[0, 0].set_xlabel('Pred. Concentrations (µg/mL)', fontsize=8)
    axs[0, 0].set_ylabel('Obs. Concentrations (µg/mL)', fontsize=8)
    axs[0, 0].set_xlim(0, max_c)
    axs[0, 0].set_ylim(0, max_c)
    axs[0, 0].tick_params(labelsize=8)

    axs[0, 1].plot(torch.range(0, max_c), torch.range(0, max_c), colors_[7])
    axs[0, 1].set_xlabel('Pred. Concentrations (µg/mL)', fontsize=8)
    axs[0, 1].set_ylabel('Obs. Concentrations (µg/mL)', fontsize=8)
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xlim(10e-2, max_c)
    axs[0, 1].set_ylim(10e-2, max_c)
    axs[0, 1].tick_params(labelsize=7, which='both')

    axs[1, 0].set_xlabel('Pred. Concentrations (µg/mL)', fontsize=8)
    axs[1, 0].set_ylabel('Error (OBS - PRED)', fontsize=8)
    axs[1, 0].set_xlim(0, max_pred)
    axs[1, 0].set_ylim(min_er, max_er)
    axs[1, 0].tick_params(labelsize=8)

    axs[1, 1].set_xlabel('Pred. Concentrations (µg/mL)', fontsize=8)
    axs[1, 1].set_ylabel('Error (OBS - PRED)', fontsize=8)
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlim(0, max_c)
    axs[1, 1].set_ylim(min_er, max_er)
    axs[1, 1].tick_params(labelsize=7, which='both')

    axs[2, 0].set_xlabel('Time (days)', fontsize=8)
    axs[2, 0].set_ylabel('Error (OBS - PRED)', fontsize=8)
    axs[2, 1].set_xlim(0, t_max)
    axs[2, 0].set_ylim(min_er, max_er)
    axs[2, 0].tick_params(labelsize=8)

    axs[2, 1].set_xlabel('Time after dose (days)', fontsize=8)
    axs[2, 1].set_ylabel('Error (OBS - PRED)', fontsize=8)
    axs[2, 1].set_xlim(0, tsld_max)
    axs[2, 1].set_ylim(min_er, max_er)
    axs[2, 1].tick_params(labelsize=8)

    fig.suptitle('GOF Plots')

    plt.tight_layout()
    plt.savefig(save_name, dpi=400)
    plt.close()
    print('GOF image saved. Epoch %d'%(epoch))


# ========================================
# =========== DIFF SAMPLINGS =============
# ========================================
@torch.no_grad()
def scatter_(real, pred, save_name, name='Conc', log=False, val=False):

    real, pred = real.numpy(), pred.numpy()
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
    sns.scatterplot(df,  x='Pred', y='Real')
    plt.ylabel('Real %s'%(name))

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

    if val:
        plt.xlabel('Mean Predicted %s'%(name))
        plt.title('Real %s vs Predicted Mean %s'%(name, name), fontsize=14)
    else:
        plt.xlabel('Predicted %s'%(name))
        plt.title('Real %s vs Predicted %s'%(name, name), fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

@torch.no_grad()
def scatter_dcolors(real, pred, save_name, name='Conc', log=False, val=False):

    real, pred = real.numpy(), pred.numpy()
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

    plt.figure()

    for i in range(real.shape[0]):
        df = pd.DataFrame({'Real': real[i], 'Pred': pred[i]})
        sns.scatterplot(df,  x='Pred', y='Real', color=colors_[i],
                        label='Pat%d'%(i)).set(xlabel=None)
    plt.ylabel('Real %s'%(name))
    if val:
        plt.xlabel('Mean Predicted %s'%(name))
        plt.title('Real %s vs Predicted Mean %s'%(name, name), fontsize=14)
    else:
        plt.xlabel('Predicted %s'%(name))
        plt.title('Real %s vs Predicted %s'%(name, name), fontsize=14)
    plt.legend(bbox_to_anchor=(1.00, -0.15), ncol=5, fancybox=True, fontsize=8) # Below
    plt.tight_layout(pad=1.5)
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

    plt.savefig(save_name, dpi=400)
    plt.close()

@torch.no_grad()
def diffsamp_mean_traj(X, pred_x, T_DV, T_ODE, epoch, save_name):

    # Information for axis
    max_x = X.max()
    max_pred = pred_x.max()
    max_c = max_x if max_x >= max_pred else max_pred

    if max_c % 5 != 0:
        max_c = (max_c//5) * 5 + 5

    t_org = T_DV.numpy()
    T_ODE = T_ODE.numpy()
    xlim = T_ODE.max()
    X_copy, pred_copy = X.clone(), pred_x.clone()
    X_copy, pred_copy = X_copy.numpy(), pred_copy.numpy()
    # X_copy has dimensions #Pat x #Rep x Time

    for i in range(X_copy.shape[0]):

        X_pat, Pred_pat = X_copy[i], pred_copy[i]
        id_, id_p, time_, time_p, r_conc, p_conc = [], [], [], [], [], []
        for r in range(X_pat.shape[0]):

            for t in range(X_pat.shape[1]):
                id_.append(r)

                time_.append(t_org[i, t])
                r_conc.append(X_pat[r, t])

            for t in range(Pred_pat.shape[1]):
                id_p.append(r)

                time_p.append(T_ODE[t])
                p_conc.append(Pred_pat[r, t])

        real_df = pd.DataFrame(
            {'ID': id_, 'Time': time_, 'Real_Conc': r_conc})
        pred_df = pd.DataFrame(
            {'ID': id_p, 'Time': time_p, 'Pred_Conc': p_conc})

        sns.scatterplot(data=real_df, x='Time', y='Real_Conc', color=colors_[i],
                        label='Obs Pat%d'%(i)).set(xlabel=None)
        sns.lineplot(data=pred_df, x='Time', y='Pred_Conc', color=colors_[i],
                     estimator=np.median, err_style='band', errorbar=('pi', 95),
                     label='Pred Pat%d'%(i)).set(xlabel=None)

    plt.xlabel('Time in days')
    plt.ylabel('Drug concentration (µg/mL)')
    plt.xlim(0, xlim)
    plt.ylim(0, max_c)
    # plt.legend(fontsize=8)
    plt.legend(bbox_to_anchor=(1.00, -0.15), ncol=5, fancybox=True, fontsize=8) # Below
    plt.tight_layout(pad=1.8) # Right and below
    plt.title('Concentration profile')
    plt.savefig(save_name, dpi=400)
    plt.close()
    print('Image saved. Epoch %d'%(epoch))


@torch.no_grad()
def Residual_plots(X, pred_x, T_data, TSLD, epoch, save_name, val=False):

    # Here X is the original data from the CSV which
    # also contains the random effect part
    t_data = T_data.clone().cpu()
    t_max = t_data.max()
    tsld = TSLD.clone().cpu()
    tsld_max = tsld.max()

    error = X - pred_x

    # Information for axis
    max_x = X.max()
    max_pred = pred_x.max()
    max_c = max_x if max_x >= max_pred else max_pred
    max_er, min_er = error.max(), error.min()

    if max_c % 5 != 0:
        max_c = (max_c//5) * 5 + 5

    if max_er % 5 != 0:
        max_er = (max_er//5) * 5 + 5

    if min_er % 5 != 0:
        min_er = (min_er//5) * 5 - 5
        
    if t_max % 4 != 0:
        t_max = (t_max//4) * 4 + 4

    if tsld_max % 4 != 0:
        tsld_max = (tsld_max//4) * 4 + 4

    fig, axs = plt.subplots(1, 2, figsize=(12, 6)) # , sharex=True, sharey=True)
    axs[0].axhline(y=0.0, color='r', linestyle='-')
    axs[1].axhline(y=0.0, color='r', linestyle='-')

    for i in range(len(X)):
        axs[0].plot(t_data[i, :], error[i, :], '.', color=colors_[0])
        axs[1].plot(tsld[i, :], error[i, :], '.', color=colors_[0])

    axs[0].set_xlabel('Time (days)', fontsize=8)
    axs[0].set_ylabel('Error (OBS - PRED)', fontsize=8)
    axs[0].set_xlim(0, t_max)
    axs[0].set_ylim(min_er, max_er)
    axs[0].tick_params(labelsize=8)

    axs[1].set_xlabel('Time after dose (days)', fontsize=8)
    axs[1].set_ylabel('Error (OBS - PRED)', fontsize=8)
    axs[1].set_xlim(0, tsld_max)
    axs[1].set_ylim(min_er, max_er)
    axs[1].tick_params(labelsize=8)

    fig.suptitle('Error Plots')

    plt.tight_layout()
    plt.savefig(save_name, dpi=400)
    plt.close()
    print('Residual plot saved. Epoch %d'%(epoch))
