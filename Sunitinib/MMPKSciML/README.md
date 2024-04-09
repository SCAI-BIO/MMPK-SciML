# MMPK-SciML

## Description

This fold contains the code for running our MMPK-SciML Model using the 5FU data.

## Run

First you need to create an anaconda environment. Please use the following command:
```
$ conda env create -f requirements.yml
$ conda activate MMPKSciML
```

You can train the model using different parameters just writing them when running the model or modifying their default values. Use -1 in GPU for CPU. Please see the parser.py script

If you just want to train the model with a learnable volume please use
```
python main.py --exp_name=$name --GPU=$gpu --fold=$fold --log_scale=0 --time_days=0 --reg_L1=0 --both=0 --norm_w_loss=Real --met_loss=0 --w_loss=1 --type_ode=MathW_ODE --batch_size=$bs --lr=$lr --num_epochs=$epochs --patience=$patience --dataset=Sunitinib --drug_adm=Achims --dim_params=4 --lstm_type=tlstm --lstm_n_layers=$nl_lstm --lstm_h_size=$hs_lstm --senc_h_size=$senc_h_size --senc_n_layers=$senc_n_layers --act_senc=$act_senc --norm_senc=$norm_senc --mlp_n_layers=$nl_mlp --act_mean=$a_m --act_var=$a_l --solver=Normal --method_solver=dopri5 --rtol=$rtol --atol=$atol --save_fig_freq=$s_fig --save_freq=$s_freq --lambda_mse=$l_mse --lambda_kl=$l_kl --lambda_mse_met=$l_mse_met --prior_mean=0.0 --prior_std=1.0
```

## Validation

val_data_split defines how the validation will be done. "Same_Patients" is for testing the model in the training data. "New_Patients" for the test patients but without sampling. "New_Patients_PostDist" valdiates the model using the test patients sampling from the approximated posterior distribution.nruns_ppd_pp is the number of samples from the approximated posterior distribution

The validation process needs the following 3 steps
```
python main.py --from_best=True --mode=val --val_data_split=$vdsp --nruns_ppd=$nruns --met_loss=0 --exp_name=$name --GPU=$gpu --fold=$fold --log_scale=0 --time_days=0 --type_ode=MathW_ODE --dataset=Sunitinib --drug_adm=Achims --dim_params=4 --lstm_type=tlstm --lstm_n_layers=$nl_lstm --lstm_h_size=$hs_lstm --senc_h_size=$senc_h_size --senc_n_layers=$senc_n_layers --act_senc=$act_senc --norm_senc=$norm_senc --mlp_n_layers=$nl_mlp --act_mean=$a_m --act_var=$a_l --solver=Normal --method_solver=dopri5 --rtol=$rtol --atol=$atol --prior_mean=0.0 --prior_std=1.0

python posterior_plots.py --from_best=True --exp_name=$name --nruns_ppd=$nruns --fold=$fold --dataset=Sunitinib --val_data_split=$vdsp --type_ode=MathW_ODE

python Pat_plots.py --from_best=True --exp_name=$name --nruns_ppd=$nruns --fold=$fold --dataset=Sunitinib --val_data_split=$vdsp --type_ode=MathW_ODE --met_loss=0
```
In case you want to validate the model using the weights of a saved epoch, please use "epoch_init=$epoch" instead of "from_best=True" 

# Fold organization
                
    ├── data                       <- Scripts to load the datasets
    │   ├── __init__.py
    │   ├── datasets.py            <- Script to create the PyTorch dataset to be used in the dataloader
    │   └── load_data.py           <- Script to load the data and datasets according to the parser
    │
    ├── requirements.yml           <- The requirements file for reproducing the analysis environment
    │
    ├── src                        <- Source code for use in this project.
    │   ├── __init__.py
    │   └── ....py          <- Main scripts to train and validate the model
--------