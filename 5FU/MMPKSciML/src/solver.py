import os
import warnings
import numpy as np
import torch
import pickle
from networks import *
from utils import RunningAverageMeter, print_network
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, dataloader, dataloader_val):

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val

        self.T1 = (torch.range(0, 18, 1)/24).to(self.device)
        self.Tval = (torch.range(0, 18, 1)/24).to(self.device)

        self.method = self.config.method_solver
        self.rtol = self.config.rtol
        self.atol = self.config.atol

        self.build_model()

    def build_model(self):

        IC = [self.config.IC_CL, self.config.IC_V]

        if self.config.type_ode == 'PPCL_PopV':
            self.config.dim_params = 1
            self.ODE = ODEFunc_PPCL_PopV(self.config, IC=IC).to(self.device)
        else:
            self.config.dim_params = 2
            self.ODE = ODEFunc_CLV_PP(self.config, IC=IC).to(self.device)

        self.Enc = MLP_Enc(self.config).to(self.device)

        print('Models are build and have set to device')
        if 'train' in self.config.mode:
            self.loss = RunningAverageMeter()
        self.get_optimizer()

        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e15

        self.print_models()
        if 'train' in self.config.mode:
            self.set_nets_train()
        else:
            self.set_nets_eval()

    def get_optimizer(self):

        params =  list(self.Enc.parameters()) + list(self.ODE.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, self.config.lr)

    def load_models(self):

        if self.config.from_best:
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Best.pth'),
                map_location=self.device)
            self.config.epoch_init = weights['Epoch']
            epoch = self.config.epoch_init
        else:
            epoch = self.config.epoch_init
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)),
                map_location=self.device)

        self.best_loss = weights['Loss']

        self.Enc.load_state_dict(weights['Enc'])
        self.ODE.load_state_dict(weights['ODE'])

        if 'train' in self.config.mode:
            self.optimizer.load_state_dict(weights['Opt'])

        print('Models have loaded from epoch:', epoch)

    def save(self, epoch, loss, best=False):

        weights = {}
        weights['Enc'] = self.Enc.state_dict()
        weights['ODE'] = self.ODE.state_dict()
        weights['Opt'] = self.optimizer.state_dict()

        weights['Loss'] = loss
        if best:
            weights['Epoch'] = epoch
            torch.save(weights, 
                os.path.join(self.config.save_path_models, 'Best.pth'))
        else:
            torch.save(weights, 
                os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        print('Models have been saved')

    def print_models(self):

        print_network(self.Enc, 'VI Encoder')
        print_network(self.ODE, 'ODE')

    def set_nets_train(self):
        self.Enc.train()
        self.ODE.train()

    def set_nets_eval(self):
        self.Enc.eval()
        self.ODE.eval()

    # ==================================================================#
    # ==================================================================#
    def get_gpu_memory_used(self):
        import GPUtil
        if torch.cuda.is_available():
            try:
                mem = int(GPUtil.getGPUs()[self.config.GPU_print[0]].memoryUsed)
            except BaseException:
                mem = 0
            if hasattr(self, 'GPU_MEMORY_USED'):
                mem = max(mem, self.GPU_MEMORY_USED)
            return mem
        else:
            return 0
