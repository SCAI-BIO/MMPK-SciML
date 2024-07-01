########################################################################################################################
# Main 1 for Suni                                                                                                      #
# Created by: Alina Pollehn and Dr. Elena Trunz, edited by Olga Teplytska                                              #
# last edited: 20.02.24  (Changed Python version for MissForest, which currently requires 3.10)                        #
# this has been written in Python 3.11 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# 'suni_data_split_0_aug_preproc.csv': preprocessed data without augmentation                                          #                                                                                                          #
# f'suni_data_split_{split}_aug_{data_augmentation}_preproc_corr.csv' with augmented data                              #
########################################################################################################################

import Suni_OtherMethods_clean
import Suni_MoreBoosters_clean
import faulthandler

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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
faulthandler.enable()

########################################################################################################################
# Split 1                                                                                                              #
########################################################################################################################

if job_array_id == 0:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="1", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="1", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="1", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="1", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="1", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="1", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)

########################################################################################################################
# Split 2                                                                                                              #
########################################################################################################################

if job_array_id == 1:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="2", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="2", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="2", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="2", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="2", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="2", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)

########################################################################################################################
# Split 3                                                                                                              #
########################################################################################################################
if job_array_id == 2:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="3", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="3", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="3", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="3", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="3", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="3", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)


########################################################################################################################
# Split 4                                                                                                              #
########################################################################################################################
if job_array_id == 3:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="4", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="4", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="4", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="4", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="4", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="4", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)

########################################################################################################################
# Split 5                                                                                                              #
########################################################################################################################
if job_array_id == 4:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="5", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="5", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="5", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="5", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="5", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="5", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)

########################################################################################################################
# Split 6                                                                                                              #
########################################################################################################################
if job_array_id == 5:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="6", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="6", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="6", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="6", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="6", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="6", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)

########################################################################################################################
# Split 7                                                                                                              #
########################################################################################################################
if job_array_id == 6:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="7", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="7", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="7", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="7", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="7", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="7", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)


########################################################################################################################
# Split 8                                                                                                              #
########################################################################################################################
if job_array_id == 7:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="8", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="8", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="8", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="8", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="8", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="8", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)


########################################################################################################################
# Split 9                                                                                                              #
########################################################################################################################
if job_array_id == 8:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="9", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="9", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="9", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="9", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="9", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="9", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)


########################################################################################################################
# Split 10                                                                                                             #
########################################################################################################################

if job_array_id == 9:

    best_params_2 = Suni_OtherMethods_clean.optimize_random_forest_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="Random_Forest",
        num_trials=100)

    best_params_4 = Suni_OtherMethods_clean.optimize_gradient_boosting_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="Gradient_Boosting",
        num_trials=100)

    best_params_6 = Suni_OtherMethods_clean.optimize_support_vector_machine_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="Support_Vector_Machine",
        num_trials=100)

    # Best parameters XGBoost, LightGBM

    best_params_8 = Suni_MoreBoosters_clean.optimize_xtreme_gradient_boosting_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="Xtreme_Gradient_Boosting",
        num_trials=100)

    best_params_10 = Suni_MoreBoosters_clean.optimize_light_gradient_boosting_hyperparameters(
        split="10", data_augmentation="100",
         algorithm_name="Light_Gradient_Boosting",
        num_trials=100)

    # Run scripts with best hyperparameters

    Suni_OtherMethods_clean.run_random_forest_experiment(split="10", data_augmentation="100",
                                                        algorithm_name="Random_Forest",
                                                       best_params=best_params_2,
                                                       num_epochs=1000)


    Suni_OtherMethods_clean.run_gradient_boosting_experiment(split="10", data_augmentation="100",
                                                           
                                                           algorithm_name="Gradient_Boosting",
                                                           best_params=best_params_4,
                                                           num_epochs=1000)

    Suni_OtherMethods_clean.run_support_vector_machine_experiment(split="10", data_augmentation="100",
                                                                
                                                                algorithm_name="Support_Vector_Machine",
                                                                best_params=best_params_6,
                                                                num_epochs=1000)



    Suni_MoreBoosters_clean.run_xtreme_gradient_boosting_experiment(split="10", data_augmentation="100",
                                                                  
                                                                  algorithm_name="Xtreme_Gradient_Boosting",
                                                                  best_params=best_params_8,
                                                                  num_epochs=1000)


    Suni_MoreBoosters_clean.run_light_gradient_boosting_experiment(split="10", data_augmentation="100",
                                                                 
                                                                 algorithm_name="Light_Gradient_Boosting",
                                                                 best_params=best_params_10,
                                                                 num_epochs=1000)
