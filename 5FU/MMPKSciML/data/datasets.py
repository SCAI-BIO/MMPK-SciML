import torch
from torch.utils.data import Dataset

class IndM_Dataset(Dataset):
    def __init__(self, config, data):

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