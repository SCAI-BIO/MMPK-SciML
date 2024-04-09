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
python main.py --conc_inp=True --norm_senc=$norm_senc --fold=$fold --type_loss=WMSE --GPU=$gpu --exp_name=$nameFixV --dataset=IndM --covariates=Basic --lr=$lr  --hh_ssize=$hh_ssize --senc_n_layers=$senc_n_layers --act_senc=$act_senc --l_act_senc=$l_act_senc  --solver=Normal --method_solver=dopri5 --type_ode=PPCL_PopV --dim_params=1 --scaler=none --trans_output=$trans_out --lambda_mse=$l_mse --lambda_kl=$l_kl --num_epochs=$epochs --patience=$patience --save_freq=$s_freq --save_fig_freq=$s_fig
```

If you just want to train the model with a fixed volume please use
```
python main.py --conc_inp=True --fix_V=True --norm_senc=$norm_senc --fold=$fold --type_loss=WMSE --GPU=$gpu --exp_name=$nameFixV --dataset=IndM --covariates=Basic --lr=$lr  --hh_ssize=$hh_ssize --senc_n_layers=$senc_n_layers --act_senc=$act_senc --l_act_senc=$l_act_senc  --solver=Normal --method_solver=dopri5 --type_ode=PPCL_PopV --dim_params=1 --scaler=none --trans_output=$trans_out --lambda_mse=$l_mse --lambda_kl=$l_kl --num_epochs=$epochs --patience=$patience --save_freq=$s_freq --save_fig_freq=$s_fig
```

In case you want to validate the model using the weights of a saved epoch, please use "epoch_init=$epoch" instead of "from_best=True" 

## Validation

val_data_split defines how the validation will be done. "Same_Patients" is for testing the model in the training data. "New_Patients" for the test patients but without sampling. "New_Patients_PostDist" valdiates the model using the test patients sampling from the approximated posterior distribution.nruns_ppd_pp is the number of samples from the approximated posterior distribution

If you trained a model with a learnable volume please use
```
python main.py --mode=val --fix_V=True --nruns_ppd_pp=$nruns_ppd_pp --fold=$fold --val_data_split=$vdsp --conc_inp=True --from_best=True --norm_senc=$norm_senc --GPU=$gpu --exp_name=$name --dataset=IndM --covariates=Basic --hh_ssize=$hh_ssize --senc_n_layers=$senc_n_layers --act_senc=$act_senc --l_act_senc=$l_act_senc --solver=$solv --method_solver=$method --type_ode=$t_ode --dim_params=1 --scaler=none
```

If you trained a model with a fixed volume please use
```
python main.py --mode=val --fix_V=True --nruns_ppd_pp=$nruns_ppd_pp --fold=$fold --val_data_split=$vdsp --conc_inp=True --from_best=True --norm_senc=$norm_senc --GPU=$gpu --exp_name=$name --dataset=IndM --covariates=Basic --hh_ssize=$hh_ssize --senc_n_layers=$senc_n_layers --act_senc=$act_senc --l_act_senc=$l_act_senc --solver=$solv --method_solver=$method --type_ode=$t_ode --dim_params=1 --scaler=none
```


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