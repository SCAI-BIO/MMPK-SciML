#!/usr/bin/ipython
import os
import numpy as np
import torch
import torch.distributions as dist
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from glob import glob
import shutil
from statistics import NormalDist
import scipy.stats as st
from parser import base_parser
import warnings
warnings.filterwarnings('ignore')

colors_ = ['#071D49', '#9CA4B6', '#004BFF', '#FF8307', 
           '#9200E6', '#1AC9A8', '#FFE664', '#D90F64',
           '#99B7FF', '#996633', '#FFFFFF']

def confidence_interval(data, confidence=0.95):
    # IC = X_mean +- z*std / sqrt(n)
    dist = NormalDist.from_samples(data)

    # Z using alpha/2 ---> (1 - confidence/2)
    # Because it is normal we can also use (1 + confidence/2)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = z * dist.stdev / np.sqrt(len(data)-1)
    return dist.mean - h, dist.mean + h

def ParamDist(Param, save_name, nbins=10, dname='Predicted', name='CL'):

    name_title = dname%(name)
    bins_pred = np.histogram_bin_edges(Param, nbins)
    plt.figure()
    sns.histplot(Param, bins=bins_pred, kde=False, color=colors_[0]).set(xlabel=None)

    # plt.title('%s %s Distribution'%(dname, name), fontsize=14)
    plt.title(name_title, fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def etaDist(Param, save_name, nbins=10, xlim=[-2, 2], dname='Predicted', name='CL'):

    name_title = dname%(name)
    bins_pred = np.histogram_bin_edges(Param, nbins)
    plt.figure()
    sns.histplot(Param, bins=bins_pred, kde=True, color=colors_[0]).set(xlabel=None)

    plt.xlim(xlim[0], xlim[1])
    plt.title(name_title, fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_name, dpi=400)
    plt.close()

def main(config):

    print('Begin')

    # ===================================
    # ======== Load the data ============
    # ===================================
    params_data = config.path_params_data
    epoch = config.epoch

    path_ = os.path.join(config.save_path_samples, 'CL_Imgs', 'Epoch_%s'%(epoch))
    if os.path.exists(path_):
        shutil.rmtree(path_)
    os.makedirs(path_, exist_ok=True)

    params_data = torch.load(params_data)
    df = pd.read_csv(config.path_popparam_data)
    df = df.drop(['KA', 'QH'], axis=1)
    df = df[df.Epoch == epoch]
    df = df.iloc[-1]
    df = df[1:]
    eta_names = df.keys()
    PopPars = np.expand_dims(df.values, 0)
    num_etas = eta_names.shape[0]

    Params, Etas, Means, Log_Var = [], [], [], []
    keys_groups = params_data.keys()
    for key_g in keys_groups:

        params_data_g = torch.exp(params_data[key_g]['Params']).numpy()
        params_data_g = np.reshape(params_data_g, (-1, num_etas))
        Params.append(params_data_g)

        if config.type_ode == 'MathW_ODE':

            weight = params_data[key_g]['Weight'].numpy()
            ASCL = (weight/70)**0.75
            ASV = (weight/70)
            ones = np.ones_like(ASCL)

            div_ = np.concatenate((ASCL, ASCL, ASV, ASV, ASCL, ASCL, ASV, ASV, ones), 1)
            div_ = np.expand_dims(div_, 1)
            div_ = np.repeat(div_, config.nruns_ppd, 1)
            div_ = np.reshape(div_, (-1, num_etas))

            params_data_g = params_data_g/div_

        etas_data_g = np.log(params_data_g/PopPars)
        Etas.append(etas_data_g)

        means_data_g = params_data[key_g]['Pred_Means'].numpy()
        Means.append(means_data_g)

        lv_data_g = params_data[key_g]['Pred_Log_Vars'].numpy()
        Log_Var.append(lv_data_g)

    Params = np.concatenate(Params)
    Etas = np.concatenate(Etas)
    Means = np.concatenate(Means)
    Log_Var = np.concatenate(Log_Var)
    Std = np.exp(0.5*Log_Var)
    total_patients = Params.shape[0]

    if total_patients < 100:
        nbins = 10
    else:
        nbins = 20

    dname = '%s'
    ename = '%s Eta'
    mname = '%s Eta (Mean)'
    stdname = '%s Eta (Cond. Std)'
    j=0

    for i in range(num_etas):
        eta_name = eta_names[i]
        p_name = '%s.png'%(eta_name)
        ep_name = 'Eta_%s.png'%(eta_name)
        mep_name = 'Mean_Eta_%s.png'%(eta_name)
        stdep_name = 'Std_Eta_%s.png'%(eta_name)

        min_ = np.abs((Etas[:, i].min() // 1)) + 1
        max_ = np.abs((Etas[:, i].max() // 1)) + 1
        min_ = -4
        max_ = 4
        r_eta = (-max_, max_) if max_ > min_ else (-min_, min_)

        save_name = os.path.join(path_, p_name)
        ParamDist(Params[:, i], save_name, nbins=nbins, dname=dname,
                  name=eta_name)
        save_name = os.path.join(path_, ep_name)
        etaDist(Etas[:, i], save_name, nbins=nbins,
                xlim=r_eta, dname=ename, name=eta_name)
        if eta_name in ['CL_S', 'V2_S', 'V2_M', 'F_M']:
            save_name = os.path.join(path_, mep_name)
            etaDist(Means[:, j], save_name, nbins=nbins, xlim=r_eta, 
                dname=mname, name=eta_name)
            save_name = os.path.join(path_, stdep_name)
            etaDist(Std[:, j], save_name, nbins=nbins, xlim=r_eta, 
                dname=stdname, name=eta_name)
            j += 1
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
    path_popparam_data = config.save_path_samples 

    name_ = 'Val_Imgs_%s'%(config.val_data_split)
    config.save_path_samples = os.path.join(config.save_path_samples, name_)

    # If there are more than 1 checkpoint, the model will do the plots
    # for all the saved checkpoints
    path_params_data = config.save_path_samples

    if config.from_best:
        opcs = glob(path_params_data + '/*.pth')
        epoch = 1
        for opc in opcs:
            epoch_opc = opc.split('/')[-1].split('Ep')[-1].split('.pth')[0]

            if int(epoch_opc) > epoch:
                epoch = int(epoch_opc)
        config.epoch = epoch
    assert config.epoch != 1
    config.path_params_data = os.path.join(
        path_params_data, 'Params_Values_Ep%d.pth'%(config.epoch))
    config.path_popparam_data = glob(path_popparam_data + '/*.csv')[-1]
    main(config)
