def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--mode',  type=str,
                        default='train',
                        choices=['train', 'val'])
    parser.add_argument('--dataset', type=str,
                        default='Sunitinib_Pazopanib')
    parser.add_argument('--GPU', type=str, default='-1',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=2)
    parser.add_argument('--train_dir', type=str,
                        default='/home/valderramanino/Pharmazie/Development/Development_Pharmazie/data')
    parser.add_argument('--save_path', type=str,
                        default='/home/valderramanino/Pharmazie/Development/Development_Pharmazie/models/MMPKSciML')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--train_val', type=str,
                        default='same', choices=['same', 'org'])
    parser.add_argument('--fold', type=int,
                        default=1)
    parser.add_argument('--val_data_type', type=str,
                        default='Extrapolation',
                        choices=['Extrapolation'])
    parser.add_argument('--val_data_split', type=str,
                        default='Same_Patients',
                        choices=['Same_Patients',
                                 'New_Patients',
                                 'New_Patients_PostDist'])
    parser.add_argument('--load_Params', type=bool,
                        default=False)
    parser.add_argument('--nruns_ppd', type=int,
                        default=5,
                        help='Number of runs for appx the PPD per patient')

    # Specific training parameters
    parser.add_argument('--solver', type=str,
                        default='Normal',
                        choices=['Adjoint', 'Normal'],
                        help='Solver for the ODE system')
    parser.add_argument('--method_solver', type=str,
                        default='dopri5',
                        choices=['dopri5', 'dopri8', 'bosh3',
                                'fehlberg2', 'adaptive_heun',
                                'euler', 'midpoint', 'rk4',
                                'explicit_adams', 'implicit_adams'
                                'RK45','RK23', 'DOP853', 'Radau',
                                'BDF', 'LSODA'],
                        help='Solver for the ODE system')
    parser.add_argument ('--rtol', type=float,
                        default=1e-7,
                        help='rtol option for the odeint solver')
    parser.add_argument('--atol', type=float,
                        default=1e-8,
                        help='atol option for the odeint solver')
    parser.add_argument('--norm_T', type=bool,
                        default=False)
    parser.add_argument('--reg_L1', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--time_days', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--log_scale', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--PD_Covariates', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--MET_Covariates', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--type_ode', type=str,
                        default='Math',
                        choices=['Math', 'MathW_ODE'])
    parser.add_argument('--lstm_type', type=str,
                        default='tlstm',
                        choices=['tlstm', 'lstm'])
    parser.add_argument('--lstm_h_size', type=int,
                        default=64)
    parser.add_argument('--lstm_n_layers', type=int,
                        default=1)
    parser.add_argument('--senc_h_size', type=int,
                        default=8)
    parser.add_argument('--senc_n_layers', type=int,
                        default=1)
    parser.add_argument('--act_senc', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_senc', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--mlp_n_layers', type=int,
                        default=1)
    parser.add_argument('--act_mean', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--act_var', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh', 'selu', 'softplus', 'sigmoid'])
    parser.add_argument('--norm_mean', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--norm_var', type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch'])
    parser.add_argument('--dim_params', type=int,
                        default=6)
    parser.add_argument('--batch_size', type=int,
                        default=20)
    parser.add_argument('--from_best', type=bool,
                        default=False)
    parser.add_argument('--drug_adm', type=str,
                        default='Achims',
                        choices=['Achims', 'Daily', 'Measur'])
    parser.add_argument('--norm_w_loss', type=str,
                        default='Pred',
                        choices=['Pred', 'Real', ''])
    parser.add_argument('--w_loss', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--met_loss', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--both_losses', type=int,
                        default=0,
                        help='0 is False and 1 is True')                        
    parser.add_argument('--lambda_mse_met', type=float,
                        default=1.0,
                        help='It is a percentage of lamba_mse') # 1.0
    parser.add_argument('--lambda_mse', type=float,
                        default=1.0) # 1.0
    parser.add_argument('--lambda_kl', type=float,
                        default=10.0)
    parser.add_argument('--lambda_reg_l1', type=float,
                        default=0.01)
    parser.add_argument('--in_dim_mlp', type=int,
                        default=2)
    parser.add_argument('--fix_KA', type=int,
                        default=0,
                        help='0 is False and 1 is True')
    parser.add_argument('--IC_KA', type=float,
                         default=0.34, # Estimate from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_QH', type=float,
                         default=80.0, # Fixed from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_CL_S', type=float,
                        default=35.7, # Estimate from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_CL_Met', type=float,
                        default=17.1, # Estimate from Yu et al
                        help='Parameter in normal scale')    
    parser.add_argument('--IC_Q_S', type=float,
                        default=0.5, # Top of the 90% CI from the PhD Thesis
                        help='Parameter in normal scale')
    parser.add_argument('--IC_Q_Met', type=float,
                        default=20.1, # Estimate from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_V2_S', type=float,
                        default=1360.0, # Estimate from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_V3_S', type=float,
                        default=588.0, # Fixed from Houk et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_V2_Met', type=float,
                        default=635.0, # Estimate from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_V3_Met', type=float,
                        default=388.0, # Estimate from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--IC_F_Met', type=float,
                        default=0.21, # Fixed from Yu et al
                        help='Parameter in normal scale')
    parser.add_argument('--prior_std', type=float,
                        default=1.0)
    parser.add_argument('--prior_mean', type=float,
                        default=0.0)           
    parser.add_argument('--weight_decay', type=float,
                        default=0.01)
    parser.add_argument('--lr', type=float,
                        default=0.01)
    parser.add_argument('--num_epochs', type=int,
                        default=250)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--save_fig_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--patience', type=int,
                        default=50)

    config = parser.parse_args()
    return config
