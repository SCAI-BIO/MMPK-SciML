import os
import warnings
import numpy as np
import torch
from networks import *
from utils import RunningAverageMeter, print_network
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, dataloader, dataloader_val):

        self.config = config
        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val

        self.method = self.config.method_solver
        self.rtol = self.config.rtol
        self.atol = self.config.atol

        self.build_model()

    def build_model(self):

        A_init_Time = self.initialize_imputation(time=True)
        A_init_Static = self.initialize_imputation(static=True)

        if self.config.type_ode == 'MathW_ODE':
            self.ODE = MathW_ODE(self.config).to(self.device)
        else:
            self.ODE = Math_ODE(self.config).to(self.device)
        self.config.dim_params = 4 # CL_S, V2_S, V2_Met, F_Met
        self.config.latent_dim = 5
        self.Enc = Encoder(self.config, A_init_Time,
                           A_init_Static).to(self.device)

        print('Models are build and have set to device')
        if 'train' in self.config.mode:
            self.get_optimizer()
            self.loss = RunningAverageMeter()

        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e15

        self.print_models()
        if 'train' in self.config.mode:
            self.set_nets_train()
        else:
            self.set_nets_eval()

    def initialize_imputation(self, time=False, static=False):

        # W matrix 
        # 1 if there is a NON-Missing value, 0 if it is a missing value
        if time:
            X, W = self.dataloader.dataset.get_XW_time()
        if static:
            X, W = self.dataloader.dataset.get_XW_static()
        
        W_A = torch.sum(W, 0)
        A = torch.sum(X * W, 0)

        # Normalization
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]

        if time:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if W_A[i, j] == 0:
                        A[i, j] = torch.sum(X[:, :, j]) / torch.sum(W[:, :, j])
                        W_A[i, j] = 1
       
        if static:
            for i in range(A.shape[0]):
                if W_A[i] == 0:
                    A[i] = torch.sum(X[:, j]) / torch.sum(W[:, j])
                    W_A[i] = 1

        # if not available, then average across all variables
        A[W_A==0] = torch.mean(X[W==1])
        return A

    def get_optimizer(self):
        params = list(self.Enc.parameters()) + list(self.ODE.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(
            params, self.config.lr,
            weight_decay=self.config.weight_decay)

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

        print('Models have been loaded from epoch:', epoch)

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

        name = '%s Encoder'%(self.config.lstm_type)
        print_network(self.Enc, name)
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
