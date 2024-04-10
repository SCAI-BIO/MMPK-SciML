########################################################################################################################
# Data Load 5FU project                                                                                                #
# Created by: Olga Teplytska                                                                                           #
# last edited: 21.02.24  (Different Feature Selection Method)                                                          #
# this has been written in Python 3.10 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used: '5fu_data_preproc_aug0_split.csv', preprocessed data without augmentation                                 #
# based on 10fold_data_5fu_fi_cyc_split_check.csv: dataset for every split with set column to be split on              #
########################################################################################################################

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

    # csv_filename = f'5fu_data_split_{split}_aug_{data_augmentation}_preproc.csv' # data with simulated patients
    csv_filename = '5fu_data_split_0_aug_preproc.csv' # data without simulated patients
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

    if feature_selection:
        # Feature selection code with XGBoosting
        xgb = XGBRegressor(n_estimators=100, seed=seed_value, objective='reg:squarederror')

        # Select features with importance greater than 0.05
        selection = SelectFromModel(xgb, threshold=0.05)
        selection.fit(X_train, y_train)

        # Get the indices of the selected features
        selected_feature_indices = selection.get_support(indices=True)

        all_original_feature_names = X_train.columns

        # Get the names of the selected features and their corresponding original feature names
        selected_original_feature_names = X_train.columns[selected_feature_indices]

        # Update X_train and X_test with selected features

        X_train = X_train[selected_original_feature_names]
        X_test = X_test[selected_original_feature_names]

        # Calculate feature importances from the selection object
        feature_importances = selection.estimator_.feature_importances_

        zipzap = pd.DataFrame(zip(all_original_feature_names, feature_importances),
                              columns=["Feature Name", "Feature Importance"])

        zipzap = zipzap[zipzap["Feature Name"].isin(selected_original_feature_names)]
        # Selected_original_feature_names zip all feature importances
        # Filter unimportant features from list

        # Save the selected feature names, original feature names, and feature importances to a CSV file
        selected_features_csv = f"5fu_selected_features_{algorithm_name}_{split}_{data_augmentation}.csv"

        zipzap.to_csv(selected_features_csv, index=True)

    return X_train, y_train, X_test, y_test


# Example usage
X_train, y_train, X_test, y_test = (
    train_test_split(split="1", data_augmentation="0", feature_selection=True,
                     algorithm_name=None))
