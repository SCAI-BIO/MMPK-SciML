########################################################################################################################
# Main 2 for 5FU                                                                                                       #
# Created by: Alina Pollehn and Dr. Elena Trunz, edited by Olga Teplytska                                              #
# last edited: 20.02.24  (Changed Python version for MissForest, which currently requires 3.10)                        #
# this has been written in Python 3.10 and may not work the same in future versions!                                 #
#                                                                                                                      #
# data used:                                                                                                           #
# 'fu_data_split_0_aug_preproc.csv': preprocessed data without augmentation  preprocessed data with set              #                                                                                                #
# f'fu_data_split_{split}_aug_{data_augmentation}_preproc_corr.csv' with augmented data                              #
########################################################################################################################

import fu_MLP_Regressor_clean

# Seed value
seed_value = 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)

# 5. Set the keras seed
import keras

keras.utils.set_random_seed(seed_value)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
########################################################################################################################
# Split 1                                                                                                              #
########################################################################################################################

if job_array_id == 1:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="1", data_augmentation="100", 
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 3:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="1", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 2                                                                                                              #
########################################################################################################################

if job_array_id == 5:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="2", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 7:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="2", data_augmentation="100",   
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 3                                                                                                              #
########################################################################################################################

if job_array_id == 9:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="3", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 11:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="3", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 4                                                                                                              #
########################################################################################################################

if job_array_id == 13:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="4", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 15:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="4", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 5                                                                                                              #
########################################################################################################################

if job_array_id == 17:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="5", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 19:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="5", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)
########################################################################################################################
# Split 6                                                                                                              #
########################################################################################################################

if job_array_id == 21:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="6", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 23:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="6", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 7                                                                                                              #
########################################################################################################################

if job_array_id == 25:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="7", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 27:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="7", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)
########################################################################################################################
# Split 8                                                                                                              #
########################################################################################################################

if job_array_id == 29:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="8", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 31:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="8", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 9                                                                                                              #
########################################################################################################################

if job_array_id == 33:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="9", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 35:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="9", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)

########################################################################################################################
# Split 10                                                                                                             #
########################################################################################################################

if job_array_id == 37:
    best_params_16 = fu_MLP_Regressor_clean.optimize_MLP_two_hidden_layers_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="MLP_two_hidden_layers",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_two_hidden_layers_experiment(split="10", data_augmentation="100",
                                                                algorithm_name="MLP_two_hidden_layers",
                                                                best_params=best_params_16,
                                                                num_epochs=1000)

if job_array_id == 39:
    best_params_18 = fu_MLP_Regressor_clean.optimize_MLP_one_hidden_layer_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="MLP_one_hidden_layer",
        num_trials=100)

    fu_MLP_Regressor_clean.run_mlp_one_hidden_layer_experiment(split="10", data_augmentation="100",
                                                               algorithm_name="MLP_one_hidden_layer",
                                                               best_params=best_params_18,
                                                               num_epochs=1000)
