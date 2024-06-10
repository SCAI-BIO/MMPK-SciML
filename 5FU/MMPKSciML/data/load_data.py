import os
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import torch
from .datasets import *

def load_data(config):
    path = os.path.join(config.train_dir, 'corrected_full_10fold_data_5fu_fi_cyc_split_check.csv')
    data_complete = pd.read_csv(path)
    data_complete = data_complete.rename(columns={'Difference_Start_End_Infusion': 'Real_TSLD'})
    data_complete = data_complete.rename(columns={'CL_new ': 'CL'})
    data_complete = data_complete.rename(columns={'AUC_new': 'AUC'})

    n_columns = data_complete.keys()
    n_fold = 'Set_Run%d'%(config.fold)

    for n_c in n_columns:
        if 'Set' in n_c and n_c != n_fold:
            data_complete.drop(n_c, inplace=True, axis=1)
    data_complete = data_complete.rename(columns={n_fold: 'Set'})

    # Converting the conc from ng/mL to mg/L
    data_complete.DV = data_complete.DV / 1000
    # We need to sort the date of taking the measurements
    size_ = len(data_complete.ID)
    aux = np.zeros(size_)

    # Spliting the date to be able to sort them in a correct way
    # needed because f.e if we have 01.12.2018 and 05.12.2015, pandas
    # does not  change the order because the day of the first one is 
    # smaller
    data_complete.insert(1, 'Sampling_Day', aux)
    data_complete.insert(1, 'Sampling_Month', aux)
    data_complete.insert(1, 'Sampling_Year', aux)

    for i in range(size_):
        date = data_complete.iloc[i].Date_Sampling
        split_ = date.split('.')
        d, m, y = split_[0], split_[1], split_[2]
        data_complete.loc[i, 'Sampling_Day'] = int(d)
        data_complete.loc[i, 'Sampling_Month'] = int(m)
        data_complete.loc[i, 'Sampling_Year'] = int(y)

    for i, id in enumerate(data_complete.ID.unique()):
        data_id = data_complete[data_complete.ID == id].copy()
        data_id = data_id.sort_values(['Sampling_Year', 'Sampling_Month', 'Sampling_Day'])

        if i== 0:
            new_data_complete = data_id.copy()
        else:
            new_data_complete = pd.concat((new_data_complete, data_id.copy()), axis=0).reset_index(drop=True) 

    data_complete = new_data_complete.copy()
    size_ = len(data_complete.ID)
    aux = np.zeros(size_)
    data_complete.insert(1, 'New_ID', aux)
    new_id = 1
    for i, id in enumerate(data_complete.ID.unique()):

        if i > 0:
            if id != l_id:
                new_id += 1
        data_complete.loc[data_complete.ID == id, 'New_ID'] = int(new_id)
        l_id = id

    aux = np.zeros(len(data_complete.New_ID))

    # In data TSLD is always be 18/24 
    data_complete.insert(1, 'TSLD', aux)
    data_complete.insert(1, 'Diff_T', aux)

    for i, id in enumerate(data_complete.New_ID.unique()):

        dates = data_complete[data_complete.New_ID == id]['Date_Sampling']

        tsld, diff = [], []
        for j, date in enumerate(dates):

            tsld.append(18/24)
            if j == 0:
                diff.append(18/24)
            else:
                d1 = dates.values[j-1]
                d2 = date

                d1 = datetime.strptime(d1, "%d.%m.%Y")
                d2 = datetime.strptime(d2, "%d.%m.%Y")

                delta = (d2 - d1).days
                diff.append(delta)
                # tsld.append(delta + (18/24))
            
        data_complete.loc[data_complete.New_ID == id, 'Diff_T'] = diff
        data_complete.loc[data_complete.New_ID == id, 'TSLD'] = tsld

    # These values can't be normalized/scaled
    cov_fixed = ['ID', 'New_ID', 'Set', 'Real_TSLD']

    # We are not using the Therapieschema nor the Zyklus as input 
    cov = ['DV', 'AMT', 'AUC', 'CL', 'WT', 'LBM_new', 'FM_new', 'BSA_new']
    cov_stat = ['ID', 'New_ID', 'Set', 'Age', 'Sex', 'HGT']

    if config.covariates == 'Basic_Lab':
        labor = ['Leukocytes', 'Neutrophiles_Abs', 'Neutrophiles', 'Erythrocytes',
                 'Haemoglobin', 'Haematocrit', 'Thrombocytes']
        cov = cov + labor
    elif config.covariates == 'Basic_Sympt':
        sympt = ['Nausea', 'Emesis', 'Fatigue', 'Diarrhoea',
                 'Neutropenia_new', 'Thrombocytopenia', 'Hand_Foot', 'Stomatitis']
        cov = cov + sympt
    elif config.covariates == 'Complete':
        labor = ['Leukocytes', 'Neutrophiles_Abs', 'Neutrophiles', 'Erythrocytes',
                 'Haemoglobin', 'Haematocrit', 'Thrombocytes']
        sympt = ['Nausea', 'Emesis', 'Fatigue', 'Diarrhoea',
                 'Neutropenia_new', 'Thrombocytopenia', 'Hand_Foot', 'Stomatitis']
        cov = cov + labor + sympt
    data = data_complete[cov]
    data = data.apply(pd.to_numeric)

    data_stat = data_complete[cov_stat]
    data_stat = data_stat.apply(pd.to_numeric).drop_duplicates(subset=['New_ID'])

    path_stats = os.path.join(config.save_path, '%s_stats.pkl')

    if config.scaler == 'minmax':
        pick_obj = {'min': data.min(), 'max': data.max()}
        path_stats = path_stats%('minmax')

        with open(path_stats, 'wb') as f:
            pickle.dump(pick_obj, f)

        data_norm = (data - data.min())/(data.max() - data.min())
        data_stat_norm = (data_stat - data_stat.min())/(data_stat.max() - data_stat.min())
    elif config.scaler == 'standard':

        pick_obj = {'mean': data.mean(), 'std': data.std()}
        path_stats = path_stats%('standard')

        with open(path_stats, 'wb') as f:
            pickle.dump(pick_obj, f)

        data_norm = (data-data.mean())/data.std()
        data_stat_norm = (data_stat-data_stat.mean())/data_stat.std()
    else:
        data_norm = data.copy()
        data_stat_norm = data_stat.copy()

    # If normalize is None, they will not be taken into account
    config.path_stats = path_stats

    data = data_complete[cov_fixed]
    data = data.apply(pd.to_numeric)

    data = pd.concat((data, data_norm[['DV', 'AMT']],
                      data_complete[['TSLD', 'Diff_T']],
                      data_norm[data_norm.columns[2:]]), axis=1)
    data_stat = pd.concat((data_stat[cov_stat[:2]], data_stat_norm[cov_stat[2:]]), axis=1)

    config.i_ssize = len(cov_stat) - 2 - 1 # Set
    # the position that AUC would be replaced by the TSLD
    # TSLD is not being helpful so far, so am not adding it
    config.i_tsize = len(cov) -1

    if not config.cl_inp:
        config.i_tsize = config.i_tsize -1 # CL

    if not config.conc_inp:
        config.i_tsize = config.i_tsize - 1 # Conc

    config.cov_l_fixed = cov_fixed + ['Diff_T', 'AUC', 'TSLD']
    config.cov_s_fixed = ['ID','New_ID', 'Set']

    if config.dataset == 'IndM':
        config.total_train_size = data_complete[data_complete.Set == 0].shape[0]
    elif config.dataset == 'RepM':
        config.total_train_size = len(data_complete[data_complete.Set == 0].New_ID.unique())

    return [data, data_stat], config

def split_patients(data):
    data, data_stat = data
    data_val = data.copy()
    data_val_stat = data_stat.copy()

    data = data[data.Set == 0]
    data_stat = data_stat[data_stat.Set == 0]

    data_val = data_val[data_val.Set == 1]
    data_val_stat = data_val_stat[data_val_stat.Set == 1]

    return [data, data_stat], [data_val, data_val_stat]

def load_dataset(config):
    data, config = load_data(config)
    if config.fold == 0:
        data_train, data_val = data, data
    else:
        data_train, data_val = split_patients(data)

    if config.dataset == 'IndM':
        dataset = IndM_Dataset(config, data_train)

        if 'Same_Patients' in config.val_data_split:
            data_val = data_train.copy() 
            dataset_val = IndM_Dataset(config, data_train, val=True)
            bs_val = len(data[0])
        else:
            dataset_val = IndM_Dataset(config, data_val, val=True)
            bs_val = len(data_val[0])

        dataloader = get_loader(config, dataset)
        dataloader_val = get_loader(config, dataset_val, bs_val)


    return config, dataloader, dataloader_val


def get_loader(config, dataset, bs=None):

    pm = True if torch.cuda.is_available() else False

    if bs is None:
        bs = config.batch_size
        shuffle = True
    else:
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = bs,
        shuffle = shuffle,
        num_workers=config.num_workers,
        pin_memory=pm,
        drop_last = False)
    return dataloader
