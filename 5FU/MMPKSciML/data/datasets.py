import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class IndM_Dataset(Dataset):
    def __init__(self, config, data, val=False):

        self.df_l = data[0]
        self.df_s = data[1]

        self.cov_l_f = config.cov_l_fixed
        self.cov_s_f = config.cov_s_fixed
        if not config.conc_inp:
            self.cov_l_f += ['DV']
        if not config.cl_inp:
            self.cov_l_f += ['CL']

    def __getitem__(self, idx):

        df_l = self.df_l.iloc[[idx]]
        df_inps_l = df_l.loc[:, ~df_l.columns.isin(self.cov_l_f)]

        num_pat = df_l.New_ID.values[0]
        df_s = self.df_s[self.df_s.New_ID == num_pat]
        df_s = df_s.loc[:, ~df_s.columns.isin(self.cov_s_f)]

        inp_t = torch.from_numpy(df_inps_l.values)
        inp_s = torch.from_numpy(df_s.values)
        inps = torch.cat((inp_t, inp_s), 1)[0].float()

        conc = torch.from_numpy(df_l['DV'].values).float()
        doses = torch.from_numpy(df_l['AMT'].values).float()
        auc = torch.from_numpy(df_l['AUC'].values).float()
        cl = torch.from_numpy(df_l['CL'].values).float()
        real_tsld = torch.from_numpy(df_l['Real_TSLD'].values).float()

        return inps, conc, doses, auc, cl, real_tsld

    def __len__(self):
        return self.df_l.shape[0]

    def get_train_samples(self, ids):

        inps_train = []
        for id in ids:
            inps_, _, _, _, _, _ = self.__getitem__(id)
            inps_train.append(inps_.unsqueeze(0))
        inps_train = torch.cat(inps_train)

        return inps_train

class Groups_Dataset(Dataset):
    def __init__(self, config, data, val=False):

        data_long = data[0]
        data_stat = data[1]
        self.config = config

        self.cov_l_f = config.cov_l_fixed
        self.cov_s_f = config.cov_s_fixed
        if not config.conc_inp:
            self.cov_l_f += ['DV']
        if not config.cl_inp:
            self.cov_l_f += ['CL']

        self.cov_l_f += ['ID_Group']
        self.cov_s_f += ['ID_Group']
        if val:
            self.bs = -1
        else:
            self.bs = config.batch_size

        keys_ = data_long.keys()

        self.data_groups = {}
        for _, key in enumerate(keys_):
            df_s = data_stat[key]
            df_l = data_long[key]
            g_ids = df_l['New_ID'].unique()
            npat_group = len(g_ids)

            aux_l = np.zeros(len(df_l.New_ID))
            df_l.insert(1, 'ID_Group', aux_l)
            aux_s = np.zeros(len(df_s.New_ID))
            df_s.insert(1, 'ID_Group', aux_s)

            new_id = 0
            for i, id in enumerate(df_l.New_ID.unique()):

                if i > 0:
                    if id != l_id:
                        new_id += 1
                df_l.loc[df_l.New_ID == id, 'ID_Group'] = int(new_id)
                df_s.loc[df_s.New_ID == id, 'ID_Group'] = int(new_id)
                l_id = id

            # df_s = df_s.loc[:, ~df_s.columns.isin(['ID','New_ID', 'ID_Group', 'Set'])]
            df_s = df_s.loc[:, ~df_s.columns.isin(self.cov_s_f)]
            pat, dim_s = df_s.shape
            Inp_s = torch.from_numpy(df_s.values).view(pat, dim_s).float()

            df_inps_l = df_l.loc[:, ~df_l.columns.isin(self.cov_l_f)]
            s_t, dim = df_inps_l.shape
            t = int(s_t/pat)
            Inp_t = torch.from_numpy(df_inps_l.values).view(pat, t, dim).float()

            DV = torch.from_numpy(df_l['DV'].values).view(pat, t).float()
            AMT = torch.from_numpy(df_l['AMT'].values).view(pat, t, 1).float()
            AUC = torch.from_numpy(df_l['AUC'].values).view(pat, t).float()
            CL = torch.from_numpy(df_l['CL'].values).view(pat, t).float()
            Diff_Time_ODE = torch.from_numpy(df_l['Diff_T'].values).view(pat, t, 1).float()
            Diff_Time = torch.from_numpy(df_l['Diff_T'].values).view(pat, t, 1).float()
            Diff_Time = Diff_Time[:, 1:]
            zeros = torch.zeros((pat, 1, 1))
            Diff_Time = torch.cat((zeros, Diff_Time), 1)
            Diff_Time_ODE = torch.cat((zeros, Diff_Time_ODE), 1)

            data_g = {
                'Num_Pat': npat_group,
                'DV':DV,
                'AMT': AMT,
                'AUC': AUC,
                'CL': CL,
                'Inp_s': Inp_s,
                'Inp_t': Inp_t,
                'Diff_Time': Diff_Time,
                'Diff_Time_ODE': Diff_Time_ODE}
            self.data_groups[key] = data_g
        self.create_batches()

    def preprocess_batches(self):
       
        for idx_b in range(len(self.batches)):
            key, idxs = self.batches[idx_b]
            group = self.data_groups[key].copy()

            AMT = group['AMT'][idxs]
            Diff_Time = group['Diff_Time_ODE'][idxs]

            time_dose_aux = self.config.diff_dose_meas / 24

            T_aux = torch.zeros_like(Diff_Time)
            T_ODE = torch.zeros_like(Diff_Time)
            TSLD_ODE = torch.zeros_like(Diff_Time)
            Doses_ODE = torch.zeros_like(Diff_Time)
            Mask_ODE = torch.zeros_like(Diff_Time)

            pat, t_, _ = Diff_Time.shape

            if t_ == 2:
                # Only one measurement
                Doses_ODE[:, 0] = AMT[:, 0]
                T_ODE = Diff_Time[0, :, 0]
                Mask_ODE[:, 1] = 1
                TSLD_ODE[:, 1] = time_dose_aux
            else:
                for i in range(1, Diff_Time.shape[1]):

                    add_ = (T_aux[:, i-1, :] + Diff_Time[:, i, :])
                    T_aux[:, i] = add_

                # 0s and 18h . Both values are the same for everybody
                # We need to have the T in which the dose is given (18 h before)
                # and the T in which the measurements are taken, and also after 24h
                # to be able we are giving the dose during a complete day
                T_ODE = T_aux[:, :2, :]
                ones_24h = torch.ones((pat, 1, 1))
                T_ODE = torch.cat((T_ODE, ones_24h), 1)
                Doses_ODE = torch.zeros_like(T_ODE)
                Doses_ODE[:, 0] = 1
                # when the measurement was done, the patient was
                # still receiving the medicament
                Doses_ODE[:, 1] = 1
                Mask_ODE = torch.zeros_like(Doses_ODE)
                Mask_ODE[:, 1] = 1

                T_aux = T_aux[:, 2:, :]
                T_aux_min = T_aux.min().item()
                T_aux_max = T_aux.max().item()
                T_aux_range = torch.arange(T_aux_min, T_aux_max + 1)

                # First dim for the time in which the patient received the dose
                # the second for the time in which the measurement was done,
                # the third one to complete the dose cycle (24 h)
                zeros = torch.zeros(pat, 3, 1)
                ones_24h = torch.ones((1, 1, 1))
                diff_24h = 1 - time_dose_aux
                T_ODE = T_ODE[0, :, 0]

                for i in T_aux_range:
                    if i in T_aux:
                        # Previous, time of begining the infusion, time of measurement,
                        # time of finishing the infusion
                        T_ODE = torch.cat((T_ODE, torch.tensor([i - time_dose_aux]),
                                           torch.tensor([i]), torch.tensor([i + diff_24h])))
                        Doses_ODE = torch.cat((Doses_ODE, zeros), 1)
                        Mask_ODE = torch.cat((Mask_ODE, zeros), 1)

                        pats_in = (T_aux == i).sum(1)==1
                        pats_in = pats_in[:, 0]

                        # -2 because it is when the measurement is taking
                        # Remember the last is the time when the infusion finishes
                        Mask_ODE[pats_in, -2] = 1

                        # -3 because it is when the infusion begins and -2 because
                        # the patient receives the medicament for 24h
                        Doses_ODE[pats_in, -3] = 1
                        Doses_ODE[pats_in, -2] = 1

                # The 1s need to be changed to the according tsld for each patient
                # TSLD_ODE = 1 - Doses_ODE.clone()
                for p in range(pat):

                    d_p_ = AMT[p]
                    d_p_ = torch.cat((d_p_, d_p_), 1)
                    d_p_ = d_p_.view(-1)
                    Doses_ODE[p][Doses_ODE[p] == 1] = d_p_

                    # for idx, t in enumerate(T_ODE):

                    #     if Doses_ODE[p, idx] == 0:
                    #         TSLD_ODE[p][idx] = T_ODE[idx] - diff_ti
                    #     else:
                    #         diff_ti = T_ODE[idx]

                # We need to doble check that there are not duplicates values
                # otherwiese the ODE will not run
                total_t = len(T_ODE)
                unique_t = T_ODE.unique()
                total_t_unique = len(unique_t)

                if total_t != total_t_unique:
                    ids_unique = []
                    for i in range(total_t - 1):
                        t_i = T_ODE[i]
                        t_i1 = T_ODE[i + 1]

                        if t_i == t_i1:
                            ids_unique.append(torch.tensor(0))
                        else:
                            ids_unique.append(torch.tensor(1))
                    
                    # the last one is always taken
                    ids_unique.append(torch.tensor(1))
                    ids_unique = torch.stack(ids_unique)

                    T_ODE = T_ODE[ids_unique == 1]
                    Doses_ODE = Doses_ODE[:, ids_unique == 1]
                    Mask_ODE = Mask_ODE[:, ids_unique == 1]

            Doses_ODE = Doses_ODE[..., 0]
            Mask_ODE = Mask_ODE[..., 0]
            d_ode = {
                'TIME_ODE': T_ODE,
                'AMT_ODE': Doses_ODE,
                'MASK_ODE': Mask_ODE}
            self.batches[idx_b].append(d_ode)


    def create_batches(self):

        keys_ = self.data_groups.keys()
        self.batches = []

        for key in keys_:
            pat = self.data_groups[key].copy()['Num_Pat']

            if self.bs == -1:  # val
                idxs = torch.arange(0, pat)
                self.batches.append([key, idxs])
            elif pat < self.bs:
                idxs = torch.randperm(pat)
                self.batches.append([key, idxs])
            else:
                idxs = torch.randperm(pat)
                splits = torch.split(idxs, self.bs)

                for _, split in enumerate(splits):
                    self.batches.append([key, split])

        self.preprocess_batches()

    def __getitem__(self, idx):

        key, idxs, d_ode = self.batches[idx]
        group = self.data_groups[key].copy()

        Diff_Time = group['Diff_Time'][idxs].float()
        DV = group['DV'][idxs].float()
        AMT = group['AMT'][idxs].float()
        AUC = group['AUC'][idxs].float()
        CL = group['CL'][idxs].float()
        Inp_s = group['Inp_s'][idxs].float()
        Inp_t = group['Inp_t'][idxs].float()

        AMT_ODE = d_ode['AMT_ODE'].float()
        MASK_ODE = d_ode['MASK_ODE'].float()
        TIME_ODE = d_ode['TIME_ODE'].float()

        return DV, AMT, Inp_s, Inp_t, Diff_Time, AMT_ODE, MASK_ODE, TIME_ODE, AUC, CL

    def __len__(self):
        return len(self.batches)