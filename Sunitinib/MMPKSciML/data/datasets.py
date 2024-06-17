import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class Achims_SUNI_Dataset(Dataset):
    def __init__(self, config, data, val=False):

        df_l = data[0]
        df_s = data[1]

        self.dim_dv = 4
        if not config.MET_Covariates:
            self.dim_dv -= 1
        if not config.PD_Covariates:
            self.dim_dv -= 2
        self.df_l = df_l.iloc[:,  df_l.columns.isin(config.cov_long)]
        self.df_s = df_s.iloc[:,  df_s.columns.isin(config.cov_stat)]
        self.config = config
        self.max_l = config.max_l
        self.max_s = config.max_s

        if val:
            self.bs = -1
        else:
            assert config.batch_size >= 20, 'You should use a batch size bigger than 20'
            self.bs = config.batch_size

        self.create_batches()

    def preprocess_batches(self):

        self.batches = []
        for idxs in self.idxs_batches:

            data_batch_l = self.df_l.copy()
            data_batch_s = self.df_s.copy()

            COVL, COVS, Mask_Loss, Time_DV = [], [], [], []

            DV, AMT, Time_DV, TSLD = [], [], [], []
            Diff_Time, Mask_Loss, PID =[], [], []

            zeros = torch.zeros(1, 1)

            idxs = idxs.numpy()
            # Could be bigger than the batch size
            npat_batch = len(idxs)

            mask_batch_l = data_batch_l['New_ID'].isin(idxs)
            data_batch_l = data_batch_l.loc[mask_batch_l]

            mask_batch_s = data_batch_s['New_ID'].isin(idxs)
            data_batch_s = data_batch_s.loc[mask_batch_s]
            data_batch_s = data_batch_s.iloc[:,  ~data_batch_s.columns.isin(self.config.cov_fixed)]
            data_batch_s = torch.from_numpy(data_batch_s.values).view(npat_batch, -1)
            data_batch_s[data_batch_s == -99] = torch.nan

            # EVID=0 and CMT=2 is Suni concentration
            # EVID=0 and CMT=3 is Met concentration
            # EVID=0 and CMT=4 is SVEGFR_2
            # EVID=0 and CMT=5 is SVEGFR_3
            data_DV = data_batch_l[data_batch_l.EVID ==0]

            # In this way to keep the randomes in the order
            for idx in idxs:

                data_idx = data_DV[data_DV.New_ID == idx].copy()
                PID_ = torch.from_numpy(data_idx.ID.values)[0]
                PID.append(PID_)
                data_idx['DV'] = data_idx['DV'].apply(pd.to_numeric)

                # DV_idx = data_idx['DV'].apply(pd.to_numeric)
                DV_idx = torch.from_numpy(DV_idx.values).view(-1, self.dim_dv)
                DV.append(DV_idx)
                Time_idx = data_idx['TIME'].unique()
                Time_idx = torch.from_numpy(Time_idx).view(-1, 1)
                assert Time_idx.size(0) == DV_idx.size(0)
                Time_DV.append(Time_idx)

                Diff_Time_idx = Time_idx[1:] - Time_idx[:-1]
                Diff_Time_idx = torch.cat((zeros, Diff_Time_idx), 0)
                Diff_Time.append(Diff_Time_idx)

                Mask_Loss_idx = torch.ones_like(DV_idx)[:, :1]
                Mask_Loss.append(Mask_Loss_idx)
                
                Time_idx = Time_idx[:, 0]

                # In this way so that we have the same amount of
                # concentratiosn and doses for the encoder
                tsld_p, amt_p = [], []
                for it, t in enumerate(Time_idx):
                    data_p = data_batch_l[data_batch_l.New_ID == idx]
                    amt_ = data_p[data_p.TIME < t.item()]
                    # Needed to do in 2 steps to avoid problems when having
                    # more info
                    amt_ = amt_[amt_.EVID == 1]
                    lt_amt = amt_.TIME.values[-1]

                    if it > 0:
                        amt_ = amt_[amt_.TIME > Time_idx[it -1].item()]

                    tsld_ = t - lt_amt
                    amt_ = torch.tensor(amt_.AMT.apply(pd.to_numeric).sum())
                    tsld_p.append(tsld_)
                    amt_p.append(amt_)

                tsld_p = torch.stack(tsld_p).unsqueeze(1)
                amt_p = torch.stack(amt_p).unsqueeze(1)
                TSLD.append(tsld_p)
                AMT.append(amt_p)

            # In this way to avoid having 1s in the mask for the loss
            # because of the 0s at the end of the sequences
            Time_DV_MaskLoss = Time_DV.copy()
            DV = pad_sequence(DV, batch_first=True)
            Time_DV = pad_sequence(Time_DV, batch_first=True)
            Diff_Time = pad_sequence(Diff_Time, batch_first=True)
            Mask_Loss = pad_sequence(Mask_Loss, batch_first=True)
            TSLD = pad_sequence(TSLD, batch_first=True)
            AMT = pad_sequence(AMT, batch_first=True)

            COVL = torch.cat((DV, AMT, TSLD, Diff_Time), -1)
            COVS = data_batch_s

            # Done in this way because the lenght of the test set could be
            # Smaller after padding and then, the Imputation Layer has problems
            if COVL.size(1) < self.max_l:

                np_, s_, d_ = COVL.size()
                diff = self.max_l - s_
                zeros_ = torch.zeros(np_, diff, d_)
                COVL = torch.cat((COVL, zeros_), 1)

                zeros_ = torch.zeros(np_, diff, 1)
                Mask_Loss = torch.cat((Mask_Loss, zeros_), 1)
                Time_DV = torch.cat((Time_DV, zeros_), 1)
                Diff_Time = torch.cat((Diff_Time, zeros_), 1)
                TSLD = torch.cat((TSLD, zeros_), 1)
                AMT = torch.cat((AMT, zeros_), 1)

                zeros_ = torch.zeros(np_, diff, DV.size(2))
                DV = torch.cat((DV, zeros_), 1)

            if COVS.size(1) < self.max_s:
                np_, d_ = self.COVS.size()
                diff = self.max_s - d_
                zeros_ = torch.zeros(np_, diff)
                COVS = torch.cat((COVS, zeros_), 1)

            # 1 if there is a NON-Missing value, 0 if it is a missing value
            MCOVL = 1 - torch.isnan(COVL).float()
            MCOVS = 1 - torch.isnan(COVS).float()

            COVL = torch.nan_to_num(COVL, nan=0.0)
            COVS = torch.nan_to_num(COVS, nan=0.0)


            time_batch = data_batch_l.TIME.unique()
            TIME_ODE, _ = torch.from_numpy(time_batch).sort()

            AMT_ODE = torch.zeros(npat_batch, len(TIME_ODE))
            # The mask is 1 when we have a value of DV otherwise 0
            MASK_ODE = torch.zeros(npat_batch, len(TIME_ODE))

            for n in range(npat_batch):
                t_dv_p = Time_DV_MaskLoss[n][:, 0]

                # For assigning the dose, we need to take care
                # We need to use the patient specific dose regimen
                id_p = idxs[n]
                d_p = data_batch_l[data_batch_l.New_ID == id_p]
                t_v = d_p.TIME.values

                for t_id, t in enumerate(TIME_ODE):

                    if t.item() in t_v:
                        dp_t = d_p[d_p.TIME == t.item()]
                        dp_t = dp_t[dp_t.EVID ==1]
                        if len(dp_t) != 0:
                            amt = int(dp_t.AMT.item())
                            AMT_ODE[n, t_id] = amt

                    if t in t_dv_p:
                        MASK_ODE[n, t_id] = 1

            PID = torch.stack(PID)
            data_batch = {
                'DV':DV[..., :1],
                'COVL':COVL,
                'MCOVL':MCOVL,
                'COVS':COVS,
                'MCOVS':MCOVS,
                'Mask_Loss':Mask_Loss,
                'Time_DV':Time_DV,
                'TIME_ODE': TIME_ODE,
                'AMT_ODE': AMT_ODE,
                'MASK_ODE': MASK_ODE,
                'PID':PID}

            self.batches.append(data_batch)

    def create_batches(self):

        num_pats = int(self.df_l.New_ID.max())
        self.idxs_batches = []
        if self.bs == -1:
            idxs = torch.arange(1, num_pats + 1)
            self.idxs_batches.append(idxs)
        else:
            idxs = torch.randperm(num_pats) + 1
            splits = torch.split(idxs, self.bs)
            l_last = len(splits[-1])
            for id, split in enumerate(splits):

                if l_last < 0 and id == len(splits) - 2:
                    split = torch.cat((split, splits[-1]))
                    self.idxs_batches.append(split)
                    break
                else:
                    self.idxs_batches.append(split)

        self.preprocess_batches()

    def __getitem__(self, idx):

        data_batch = self.batches[idx]
        DV = data_batch['DV'].float()
        COVL = data_batch['COVL'].float()
        MCOVL = data_batch['MCOVL'].float()
        COVS = data_batch['COVS'].float()
        MCOVS = data_batch['MCOVS'].float()
        MLoss = data_batch['Mask_Loss'].float()
        TIME_ODE = data_batch['TIME_ODE'].float()
        AMT_ODE = data_batch['AMT_ODE'].float()
        MASK_ODE = data_batch['MASK_ODE'].float()
        Time_DV = data_batch['Time_DV'].float()
        PID = data_batch['PID'].float()

        return (DV, COVL, MCOVL, COVS, MCOVS,
                MLoss, AMT_ODE, MASK_ODE,
                TIME_ODE, Time_DV, PID)

    def __len__(self):
        return len(self.batches)

    def get_XW_time(self):
        # Diff Time ist not a covariate. It is used only for  
        # updating the TLSTM memory
        # return self.batches[0]['COVL'][..., :-1], self.batches[0]['MCOVL'][..., :-1]
        return self.batches[0]['COVL'], self.batches[0]['MCOVL']

    def get_XW_static(self):
        return self.batches[0]['COVS'], self.batches[0]['MCOVS']