
import os
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import torch
from .datasets import *

def load_data_achims(config):
    path = os.path.join(config.train_dir, 'Corrected_OnlySuni_Final_GemeinsameKov_Sunitinib.csv')
    data_complete = pd.read_csv(path)

    n_columns = data_complete.keys()
    n_fold = 'Set_%d'%(config.fold)

    # TRTM=2 is pazopanib
    data_complete = data_complete[data_complete.TRTM==1]

    for n_c in n_columns:
        if 'Set' in n_c and n_c != n_fold:
            data_complete.drop(n_c, inplace=True, axis=1)
    data_complete = data_complete.rename(columns={n_fold: 'Set'})

    # Converting the times to days
    if config.time_days:
        data_complete.TIME /= 24

    cov_fixed = ['New_ID', 'ID', 'Set', 'TIME']
    cov_long = ['New_ID', 'ID', 'Set', 'TIME', 'AMT', 'DV', 'EVID', 'CMT']
    cov_stat = ['New_ID', 'ID', 'Set', 'SEX', 'AGE', 'WEIGHT', 'HEIGHT', 'BSA']

    data_train_, data_val_ = split_patients(data_complete)

    size_ = len(data_train_.Patient_ID)
    aux = np.zeros(size_)
    data_train_.insert(1, 'New_ID', aux)

    new_id = 1
    for i, id in enumerate(data_train_.Patient_ID.unique()):

        if i > 0:
            if id != l_id:
                new_id += 1
        data_train_.loc[data_train_.Patient_ID == id, 'New_ID'] = int(new_id)
        l_id = id

    size_ = len(data_val_.Patient_ID)
    aux = np.zeros(size_)
    data_val_.insert(1, 'New_ID', aux)

    new_id = 1
    for i, id in enumerate(data_val_.Patient_ID.unique()):

        if i > 0:
            if id != l_id:
                new_id += 1
        data_val_.loc[data_val_.Patient_ID == id, 'New_ID'] = int(new_id)
        l_id = id

    data_train = data_train_[cov_long].copy()
    data_train_stat = data_train_[cov_stat]
    data_train_stat = data_train_stat.apply(pd.to_numeric).drop_duplicates(subset=['New_ID'])

    data_val = data_val_[cov_long].copy()
    data_val_stat = data_val_[cov_stat]
    data_val_stat = data_val_stat.apply(pd.to_numeric).drop_duplicates(subset=['New_ID'])

    data_long_full = pd.concat([data_train.copy(), data_val.copy()])
    max_ndv = 0
    for i, id in enumerate(data_long_full.ID.unique()):
        data_i = data_long_full[data_long_full.ID == id]
        data_i = data_i[data_i.EVID==0]
        data_i = data_i[data_i.CMT==2]
        m_ndv = len(data_i)

        if m_ndv > max_ndv:
            max_ndv = m_ndv

    config.max_l = max_ndv
    config.max_s = len(cov_stat) - 3

    config.i_ssize = len(cov_stat) - 3  # New_ID, ID, Set
    # DV has info of 4 variables (Suni, Meta, 2 Cov)
    # +1 to cover the complete amount because EVID and CMT are not
    # input variables
    # +2 also because of the tsld and DiffT
    config.i_tsize = len(cov_long) - len(cov_fixed) + 1 + 2  # 
    config.cov_fixed = cov_fixed
    config.cov_long = cov_long
    config.cov_stat = cov_stat

    return [data_train, data_train_stat], [data_val, data_val_stat], config

def split_patients(data):

    if type(data) == list:
        data, data_stat = data
        data_val = data.copy()
        data_val_stat = data_stat.copy()

        data = data[data.Set == 0]
        data_stat = data_stat[data_stat.Set == 0]

        data_val = data_val[data_val.Set == 1]
        data_val_stat = data_val_stat[data_val_stat.Set == 1]

        return [data, data_stat], [data_val, data_val_stat]
    else:
        data_val = data.copy()
        data = data[data.Set == 0]
        data_val = data_val[data_val.Set == 1]
        return data, data_val

def load_dataset(config):

    data_train, data_val, config = load_data_achims(config)
    if 'Same_Patients' in config.val_data_split:
        data_val = data_train.copy()

    dataset = Achims_SUNI_Dataset(config, data_train)
    dataset_val = Achims_SUNI_Dataset(config, data_val, val=True)

    dataloader = get_loader_dtp(config, dataset)
    dataloader_val = get_loader_dtp(config, dataset_val, shuffle=False)

    return config, dataloader, dataloader_val

def get_loader_dtp(config, dataset, shuffle=True):
    pm = True if torch.cuda.is_available() else False
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 1,
        shuffle = shuffle,
        num_workers=config.num_workers,
        pin_memory=pm,
        drop_last = False)
    return dataloader