import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==================================================================#
# ==================================================================#
def log_normal_pdf(data, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(data.device)
    const = torch.log(const)
    return -.5 * (const + logvar + ((data - mean) ** 2. / torch.exp(logvar)))

# ==================================================================#
# ==================================================================#
class RunningAverageMeter(object):
    # Computes and stores the average and current values
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

# ==================================================================#
# ==================================================================#
def mape_smape_diffsamp(x, pred_x, mask):
    mape, smape, rmse = [], [], []

    b = x.size(0)
    for id_ in range(b):
        mask_id = mask[id_]
        x_id_ = x[id_][mask_id == 1]
        pred_id_ = pred_x[id_][mask_id == 1]
        if (x_id_ < 1e-3).sum() > 0:
            mask_ = (x_id_ < 1e-3).float()
            x_id_[mask_ == 1] += 1e-3
        mape.append(torch.abs((x_id_ - pred_id_)/x_id_).mean())
        smape.append((torch.abs(x_id_-pred_id_) / (torch.abs(x_id_) + torch.abs(pred_id_))).mean())
        rmse.append(torch.sqrt(((x_id_ - pred_id_)**2).mean()))

    mape = torch.stack(mape)*100
    smape = torch.stack(smape)*100
    rmse = torch.stack(rmse)

    return mape, smape, rmse

def save_metrics(epoch, mape, smape, rmse, save_path):

    save_path = os.path.join(save_path, 'metrics.csv')

    metrics = [epoch, mape, smape, rmse]

    import csv
    typ = 'a' if os.path.exists(save_path) else 'wt'
    header = ['Epoch', 'MAPE', 'SMAPE', 'RMSE']
    with open(save_path, typ, encoding='UTF8') as file:

        writer = csv.writer(file)
        if typ == 'wt':
            writer.writerow(header)

        writer.writerow(metrics)

    file.close()

# ==================================================================#
# ==================================================================#

def denormalize(x, xmin, xmax):
    x = x*(xmax - xmin) + xmin
    return x

# ==================================================================#
# ==================================================================#

def expand_t(T, n):

    T_ = torch.cat(n * [T.clone().unsqueeze(0).unsqueeze(-1)], 0)
    return T_

# ==================================================================#
# ==================================================================#

def fix_pred(pred, T_ode, T_data):

    fixed_pred = []

    for idx, T_O in enumerate(T_ode):
        if T_O in T_data:
            fixed_pred.append(pred[:, idx, :])
    fixed_pred = torch.stack(fixed_pred, 1)
    return fixed_pred

# ==================================================================#
# ==================================================================#

def load_popmeans(config):
    c_exp = config.exp_name
    main_path = config.save_path.replace(c_exp, '')
    name_ = 'CPU_PK_SciML_%dPat'%(config.train_size)
    path = os.path.join(main_path, name_, 'samples', 'CL_V.csv')

    df = pd.read_csv(path)
    df = df.iloc[-1]
    mean_CL = df['CL']
    mean_V = df['V']

    return mean_CL, mean_V


# ==================================================================#
# ==================================================================#
def define_logs(config):
    import os
    import time
    typ = 'a' if os.path.exists(config.save_path_losses) else 'wt'
    with open(config.save_path_losses, typ) as opt_file:
        now = time.strftime("%c")
        opt_file.write('================ Training Loss (%s) ================\n' % now)
    
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    opt_name = os.path.join(config.save_path, 'train_opt.txt')
    with open(opt_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


# ==================================================================#
# ==================================================================#
def print_current_losses(epoch, iters, losses, t_epoch, t_comp,
                         log_name, s_excel=True):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    t_epoch = human_format(t_epoch)
    t_comp = human_format(t_comp)
    # 
    if isinstance(t_epoch, float)  and isinstance(t_comp, float):
        message = '[epoch: %d], [iters: %d], [epoch time: %.3f], [total time: %.3f] ' % (epoch, iters, t_epoch, t_comp)
    elif isinstance(t_epoch, float) and isinstance(t_comp, str):
        message = '[epoch: %d], [iters: %d], [epoch time: %.3f], [total time: %s] ' % (epoch, iters, t_epoch, t_comp)
    elif isinstance(t_epoch, str) and isinstance(t_comp, float):
        message = '[epoch: %d], [iters: %d], [epoch time: %s], [total time: %.3f] ' % (epoch, iters, t_epoch, t_comp)
    else:
        message = '[epoch: %d], [iters: %d], [epoch time: %s], [total time: %s] ' % (epoch, iters, t_epoch, t_comp)

    for k, v in losses.items():
        try:
            message += '%s=%.6f, ' % (k, v)
        except:
            message += '%s=%s, ' % (k, v)

    message = message[:-2]
    print(message)  # print the message
    with open(log_name, 'a') as log_file:
        log_file.write('%s\n' % message)  # save the message

    if s_excel:
        # Save the losses in an excel sheet and plot in an image
        excel_name = log_name[:-3] + 'xlsx'
        data = {}

        # It is necessary to do 2 times this for
        # to get the complete vectors for the graphs
        if os.path.exists(excel_name):
            loss_df = pd.read_excel(excel_name, index_col=0)
            for k in loss_df.keys():
                vect = loss_df[k].values.tolist()
                if k == 'Epoch':
                    vect.append(epoch)
                elif 'GPU' in k:
                    pass
                else:
                    vect.append(losses[k])
                data[k] = vect
        else:
            data['Epoch'] = epoch
            for k, v in losses.items():
                if 'GPU' in k:
                    pass
                else:
                    data[k] = [v]

        df = pd.DataFrame(data)
        df.to_excel(excel_name)

        if epoch > 1:
            for k in loss_df.keys():
                if k == 'Epoch':
                    time_vect = loss_df[k].values.tolist()
                elif 'GPU' in k:
                    pass
                else:
                    vect = loss_df[k].values.tolist()           
                    plt.plot(time_vect, vect, label=k, linewidth=2)

            img_name = log_name[:-3] + 'png'
            plt.legend()
            plt.title('Losses')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(img_name)
            plt.close()

# ==================================================================#
# ==================================================================#
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if magnitude == 0:
        # return str(num)
        return num
    else:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# ==================================================================#
# ==================================================================#
def print_network(model, name):

    if torch.cuda.device_count() > 1:
        model = model.module

    submodel = [(name, model)]

    for name, model in submodel:
        num_params = 0
        num_learns = 0
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                num_learns += p.numel()

        print("{} number of parameters (TOTAL): {}\t(LEARNABLE): {}.".format(
            name.upper(), human_format(num_params), human_format(num_learns)))
