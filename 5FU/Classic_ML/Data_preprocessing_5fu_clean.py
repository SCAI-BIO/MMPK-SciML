########################################################################################################################
# Data preprocessing 5FU project                                                                                       #
# Created by: Olga Teplytska                                                                                           #
# last edited: 14.02.24 (Fixed Scaling)                                                                                #
# this has been written in Python 3.11 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# corrected_10fold_5fu_clean_check.csv: clean dataset for every split with set column to be split on                   #
#                                                                                                                      #
########################################################################################################################

import pandas as pd

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

data = pd.read_csv("corrected_10fold_5fu_clean_check.csv", delimiter=',', encoding="utf-8") # clinical data
# data = pd.read_csv("corrected_10fold_5fu_clean_check_aug_1.csv", delimiter=',', encoding="utf-8") # augmented data split 1

                     
# drop times, blood counts and symptoms (proved to be not useful in previous analyses)
columns_to_drop = ['Set_Run0','Age_60', 'Therapy', 'Date_Infusion_Start', 'Time_Infusion_Start', 'Date_Infusion_End',
                   'Time_Infusion_End', 'Date_Sampling', 'Time_Sampling', 'CL_new', 'AUC_new', 'Indication',
                   'Leukocytes', 'Neutrophils_Abs', 'Neutrophils', 'Erythrocytes', 'Haemoglobin',
                   'Haematocrit', 'Thrombocytes', 'Nausea', 'Emesis', 'Fatigue', 'Diarrhoea',
                   'Neutropenia_new', 'Thrombocytopenia', 'Hand_Foot', 'Stomatitis', 'Cycle']

# convert DV to mg/L for consistency 
data['DV'] = data['DV'] / 1000

data = data.drop(columns=columns_to_drop)

# Save preprocessed data
filename = '5fu_data_split_0_aug_preproc.csv' # clinical data
# filename = '5fu_data_split_10_aug_100_preproc_corr.csv' # augmented data split 1
data.to_csv(filename, index=False)
