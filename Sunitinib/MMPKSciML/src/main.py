#!/usr/bin/ipython
import os
import warnings
import numpy as np
import torch
from parser import base_parser
from utils import define_logs
from train import Train
from train_achims import Train_Achims
import pyro
import sys
sys.path.append('../')
from data.load_data import load_dataset
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

def main(config):

    print('Mode:', config.mode)

    config, dataloader, dataloader_val = load_dataset(config)
    if config.drug_adm == 'Achims':
        Train_Achims(config, dataloader, dataloader_val)
    else:
        Train(config, dataloader, dataloader_val)

if __name__ == '__main__':

    config = base_parser()
    if config.GPU != '-1':
        config.GPU_print = [int(config.GPU.split(',')[0])]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in range(len(config.GPU.split(',')))]
    else:
        config.GPU = False

    if config.met_loss:
        assert config.MET_Covariates, 'For this loss function you need to have the metabolite as input'

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    pyro.set_rng_seed(config.seed)

    config.train_dir = os.path.join(config.train_dir, config.dataset)

    config.save_path = os.path.join(config.save_path, config.dataset,
                                    config.exp_name, 'Fold_%d'%(config.fold))
    config.save_path_samples = os.path.join(config.save_path, 'samples')
    config.save_path_models = os.path.join(config.save_path, 'models')
    config.save_path_losses = os.path.join(config.save_path, 'losses')

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.save_path_samples, exist_ok=True)
    os.makedirs(config.save_path_models, exist_ok=True)
    os.makedirs(config.save_path_losses, exist_ok=True)

    config.save_path_losses = os.path.join(config.save_path_losses, 'losses.txt')

    # Print the parser options of the current experiment
    # in a txt file saved in repo/models/exp_name/log.txt
    if 'train' in config.mode:
        define_logs(config)

    # If we are validating we run the extrapolation scenario as first step
    # therefore we can do this manually
    if config.mode == 'val' and config.val_data_type != 'Extrapolation':
        config.load_CL_V=True

    main(config)
