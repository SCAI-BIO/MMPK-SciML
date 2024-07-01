########################################################################################################################
# Data Load Suni project                                                                                               #
# Created by: Olga Teplytska                                                                                           #
# last edited: 11.06.24 (Final correction data)                                                                        #
# this has been written in Python 3.10 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used: csv_filename = 'suni_data_split_0_aug_preproc.csv' # data without simulated patients                      #
# based on:                                                                                                            #
# Datensatz_AF_FK_Sunitinib_final_raw.csv: dataset for every split with set column to be split on, corrected           #
########################################################################################################################

# for training data augmentation, encode the percentage of augmented data you want to use in 'data_augmentation="0"' in the examplary use
# at the bottom of this document

import pandas as pd
import csv
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

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

# Define real and simulated patients, split dataset based on splitting variable

def train_test_split(split="1", data_augmentation="0", feature_selection=False,
                     algorithm_name=None):
    X_train, y_train, X_test, y_test = None, None, None, None,
    selected_feature_names = []  # Initialize empty list
    selected_original_feature_names = []  # Initialize empty list

    # csv_filename = f'suni_data_split_{split}_{data_augmentation}_aug_preproc.csv' # data with simulated patients
    csv_filename = 'suni_data_split_0_aug_preproc.csv' # data without simulated patients

    data = pd.read_csv(csv_filename)

    # Identify real patients and artificial patients
    real_patients = data[data['ID'] < 200]
    artificial_patients = data[data['ID'] >= 200]
    artificial_patients.reset_index(inplace=True)

    # Data Augmentation based on specified percentage
    augmentation_percentage = int(data_augmentation)

    artificial_patients_select = artificial_patients.iloc[
                                 :int((len(artificial_patients) * (augmentation_percentage / 100)))]

    data_select = pd.concat([real_patients, artificial_patients_select])
    # all Set variables other than the one called need to be dropped
    run_to_drop = []
    for i in range(1, 11):
        run_to_drop.append(f"""Set_{i}""")
    run_to_drop.remove(f"""Set_{split}""")
    cols_to_drop = run_to_drop + ['index', 'ID']
    data_select = data_select.drop(cols_to_drop, axis=1)

    columns_to_drop2 = [f'Set_{split}']

    train_data = data_select[data_select[f'Set_{split}'] == 0]
    train_data = train_data.drop(columns_to_drop2, axis=1)
    test_data = data_select[data_select[f'Set_{split}'] == 1]
    test_data = test_data.drop(columns_to_drop2, axis=1)

    # Extract features and labels
    X_train = train_data.drop('DV', axis=1)
    y_train = train_data[['DV']]
    X_test = test_data.drop('DV', axis=1)
    y_test = test_data[['DV']]

    return X_train, y_train, X_test, y_test


# Example usage
X_train, y_train, X_test, y_test = (
    train_test_split(split="1", data_augmentation="0",
                     algorithm_name=None))
