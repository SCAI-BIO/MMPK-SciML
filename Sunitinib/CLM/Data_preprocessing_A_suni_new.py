########################################################################################################################
# Data preprocessing Suni project Augmentation                                                                         #
# Created by: Olga Teplytska                                                                                           #
# last edited: 14.02.24 (Fixed Scaling)                                                                                #
# this has been written in Python 3.11 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# Datensatz_AF_FK_Sunitinib_final_c.csv: dataset for every split with set column to be split on, corrected             #
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

data = pd.read_csv("Datensatz_AF_FK_Sunitinib_final_c.csv", delimiter=',', encoding="utf-8")

# drop ID ET0800 (no measurements), Baseline measurements and BQL measurements 
# Drop rows where 'DV' column is empty, NaN, or None
data = data.drop(data[
                     (data['PATIENT_ID'] == 'ET0800') |
                     (data['CYC'] == 'BASELINE') |
                     (data['DV'] == 'BQL') |
                     (pd.isna(data['DV']))].index).reset_index(drop=True)

# sort remaining data
data['DAT_MEAS'] = pd.to_datetime(data['DAT_MEAS'],
                                  format='%d.%m.%Y')  # convert to datetime format to be recognized by pandas
data = data.sort_values(by=['ID', 'DAT_MEAS'])  # sort by date and ID

# convert some variables to numbers which are falsely strings
data['DOS'] = pd.to_numeric(data['DOS'], errors='coerce')
data['DV_MET'] = pd.to_numeric(data['DV_MET'], errors='coerce')

# convert dose to ng for consistency
data['DOS'] = data['DOS'] * 1000

# drop columns to drop permanently: times, cycle and all SNP (single nucleotide polymorphisms)
# and body surface area and blood counts, because there is too much missing data
columns_to_permanently_drop = ['PATIENT_ID', 'CYC', 'DAT_MEAS', 'STUDY', 'TIME_MEAS', 'DAT_INTAKE', 'TIME_INTAKE',
                               'TSB', 'SMOKE', 'BSA',
                               'DAT_BP', 'TIME_BP', 'BP_SYS', 'BP_DIA', 'PULS',
                               'ALT', 'AST', 'ALB', 'ALKP', 'CAL', 'SCREA', 'LDH', 'BILI', 'CYP1', 'ABCR1',
                               'ABCR2', 'ABCR3', 'VEGFA1', 'VEGFA2', 'VEGFA3', 'VEGFA4',
                               'KDR1', 'FLT1', 'FLT2', 'FLT3', 'IL8', 'HAPAB', 'HAPVG']

data = data.drop(columns=columns_to_permanently_drop)

# Save preprocessed data
filename = 'suni_data_split_0_aug_preproc.csv'
data.to_csv(filename, index=False)
