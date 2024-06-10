def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--mode',  type=str,
                        default='train',
                        choices=['train', 'val'])
    parser.add_argument('--dataset', type=str,
                        default='IndM',
                        choices=['IndM'],
                        help='Individual Measurements')
    parser.add_argument('--covariates', type=str,
                        default='Basic',
                        choices=['Basic', 'Basic_Lab',
                                 'Complete'])
    parser.add_argument('--fold', type=int,
                        default=1)
    parser.add_argument('--GPU', type=str, default='-1',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--train_dir', type=str,
                        default='/home/valderramanino/Pharmazie/Development/Development_Pharmazie/data')
    parser.add_argument('--save_path', type=str,
                        default='/home/valderramanino/Pharmazie/Development/Development_Pharmazie/models/MMPKSciML/5FU')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--val_data_split', type=str,
                        default='Same_Patients',
                        choices=['Same_Patients',
                                 'New_Patients',
                                 'New_Patients_PostDist'])
    parser.add_argument('--load_CL_V', type=bool,
                        default=False)
    parser.add_argument('--train_test', type=bool,
                        default=False)
    parser.add_argument('--nruns_ppd_pp', type=int,
                        default=100,
                        help='Number of runs for appx the PPD per patient')
    parser.add_argument('--scaler', type=str,
                        default='none',
                        choices=['manual', 'standard', 'minmax', 'none'])
    parser.add_argument('--nbins', type=int,
                        default=4,
                        help='Number of bins for the VPC')

    # Specific training parameters
    parser.add_argument('--solver', type=str,
                        default='Adjoint',
                        choices=['Adjoint', 'Normal'],
                        help='Solver for the ODE system')
    parser.add_argument('--method_solver', type=str,
                        default='dopri5',
                        choices=['dopri5', 'dopri8', 'bosh3',
                                'fehlberg2', 'adaptive_heun',
                                'implicit_adams'],
                        help='Solver for the ODE system')
    parser.add_argument ('--rtol', type=float,
                        default=1e-7,
                        help='rtol option for the odeint solver')
    parser.add_argument('--atol', type=float,
                        default=1e-8,
                        help='atol option for the odeint solver')
    parser.add_argument('--type_ode', type=str,
                        default='PPCL_PopV',
                        choices=['PPCL_PopV', 'PPCLV'])
    parser.add_argument('--infusion_time', type=float,
                        default=24.0,
                        help='please use 0.0 for assuming a direct infusion. The value is in hours')
    parser.add_argument('--diff_dose_meas', type=float,
                        default=18.0,
                        help='Difference in time between the dose and the measurements')
    parser.add_argument('--cl_inp', type=bool,
                        default=False)
    parser.add_argument('--conc_inp', type=bool,
                        default=False)
    parser.add_argument('--num_IC', type=int,
                        default=1)
    parser.add_argument('--i_tsize', type=int,
                        default=10)
    parser.add_argument('--h_tsize', type=int,
                        default=64)
    parser.add_argument('--tenc_n_layers', type=int,
                        default=1)
    parser.add_argument('--i_ssize', type=int,
                        default=3)
    parser.add_argument('--hh_ssize', type=int,
                        default=64)
    parser.add_argument('--h_ssize', type=int,
                        default=64)
    parser.add_argument('--h_odesize', type=int,
                        default=75)
    parser.add_argument('--senc_n_layers', type=int,
                        default=1)
    parser.add_argument('--act_senc', type=str,
                        default='softplus',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--l_act_senc', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_senc', type=str,
                        default='instance',
                        choices=['instance', 'batch', 'none'])
    parser.add_argument('--i_ode', type=int,
                        default=1)
    parser.add_argument('--act_odenet', type=str,
                        default='tanh',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('-norm_ode', type=str,
                        default='instance',
                        choices=['instance', 'batch', 'none'])
    parser.add_argument('--odenet_n_layers', type=int,
                        default=2)
    parser.add_argument('--mlp_n_layers', type=int,
                        default=1)
    parser.add_argument('--act_mean', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--act_var', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--trans_output', type=str,
                        default='none',
                        choices=['none', 'log', 'arcsinh'])
    parser.add_argument('--dim_params', type=int,
                        default=2,
                        help='CL and V')
    parser.add_argument('--batch_size', type=int,
                        default=100)
    parser.add_argument('--from_best', type=bool,
                        default=False)
    parser.add_argument('--type_loss', type=str,
                        default='MSE',
                        choices=['MSE', 'L1', 'WMSE'])
    parser.add_argument('--lambda_kl', type=float,
                        default=1.0)
    parser.add_argument('--lambda_mse', type=float,
                        default=1.0)
    parser.add_argument('--IC_CL', type=float,
                        default=90.0)   
    parser.add_argument('--IC_V', type=float,
                        default=40.0)
    parser.add_argument('--fix_V', type=bool,
                        default=False)
    parser.add_argument('--lr', type=float,
                        default=0.01)
    parser.add_argument('--num_epochs', type=int,
                        default=250)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--prior_std', type=float,
                        default=1.0)
    parser.add_argument('--prior_mean', type=float,
                        default=0.0)  
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--save_fig_freq', type=int, default=20)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--patience', type=int,
                        default=50)

    config = parser.parse_args()
    return config
