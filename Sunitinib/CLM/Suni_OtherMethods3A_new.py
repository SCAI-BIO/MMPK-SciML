########################################################################################################################
# SVR, Gradient Boosting and Random Forest for Sunitinib project                                                       #
# Originally created by: Alina Pollehn and Dr. Elena Trunz, edited by Olga Teplytska                                   #
# last edited: 20.02.24  (Changed Python version for MissForest, which currently requires 3.10)                        #
# this has been written in Python 3.10 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# 'suni_data_preproc_aug0_split.csv': preprocessed data without augmentation,  with column to be split on              #
#                                                                                                                      #
# general note: cross_val_score needs negative metrics (like neg. MSE)                                                 #
# because it expects that higher values are better. However, we want to report positive metrics; therefore             #
# before saving the metric, we need a (-) sign.                                                                        #
########################################################################################################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import Data_loadA2_suni
import pandas as pd
import sklearn.neighbors._base
import sys

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, KFold
import csv
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


# Metrics

def mape_array(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(absolute_percentage_errors)


def calculate_metrics(y_true, y_pred):
    mape = mape_array(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return mape, rmse, mae


def preprocessing_pipeline(x_train):

    # Define the pipeline: Imputation with MissForest, Scaling with MinMaxScaler, One-hot encoding with
    # OneHotEncoder
    imp, scaler, onehot = MissForest(), MinMaxScaler(), OneHotEncoder()

    # Select columns
    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = make_pipeline(imp, scaler)
    cat_pipeline = make_pipeline(onehot)

    # Create a ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ], remainder='passthrough'  # pass through any unspecified columns
    )

    return preprocessor


def save_results_to_csv(results_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = results_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write the header
        writer.writeheader()
        # Write the data
        writer.writerow(results_dict)

########################################################################################################################
# Random forests
# 1. with Feature Selection
# 2. without Feature Selection
########################################################################################################################

def optimize_random_forest_hyperparameters(split, data_augmentation,
                                           feature_selection, algorithm_name,
                                           num_trials):
    # Create Optuna study for hyperparameter tuning; maximize the negative error
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_loadA2_suni.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Create a Random Forest with the suggested hyperparameters
        rgr = RandomForestRegressor(random_state=seed_value, **{
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_features': trial.suggest_categorical('max_features', [0.1, 0.5, 0.9, 'sqrt', 'log2', None]),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        })

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('rgr', rgr)])

        # Perform cross-validation to estimate the model's performance
        # test_size=1/K=20%
        cv = KFold(n_splits=5, random_state=seed_value, shuffle=True)
        mse_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

        # Calculate the mean score and save it
        mean_mse = -mse_scores.mean()

        results_dict = {
            'Tuning_MSE': mean_mse}
        csv_filename = f'suni_tuning_mse_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_random_forest_experiment(split, data_augmentation,
                                 feature_selection, algorithm_name,
                                 best_params, num_epochs):
    for i in tqdm(range(num_epochs), desc="Epochs"):
        X_train, y_train, X_test, y_test = Data_loadA2_suni.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name)

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Random Forest with the best parameter values from tuning
        regr = RandomForestRegressor(random_state=seed_value, criterion='squared_error', **best_params)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('regr', regr)])

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Calculate MAPE, RMSE and MAE for the test data
        prediction = pipeline.predict(X_test)
        y_test = np.ravel(y_test)
        prediction = np.ravel(prediction)

        mape, rmse, mae = calculate_metrics(
            y_test, prediction)

        # Create a dictionary to store the results
        results_dict = {
            'n_estimators': best_params['n_estimators'],
            'max_features': best_params['max_features'],
            'max_depth': best_params['max_depth'],
            'min_samples_split': best_params['min_samples_split'],
            'min_samples_leaf': best_params['min_samples_leaf'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

        csv_filename = f'suni_params_test_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        # Save prediction results to CSV
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'suni_results_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv',
            index=False)

########################################################################################################################
# Gradient Boosting
# 1. without Feature Selection
########################################################################################################################

def optimize_gradient_boosting_hyperparameters(split, data_augmentation,
                                               feature_selection, algorithm_name,
                                               num_trials):
    # Create Optuna study for hyperparameter tuning; maximize the negative error
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_loadA2_suni.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Create a Random Forest with the suggested hyperparameters
        gbr = GradientBoostingRegressor(random_state=seed_value, **{
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 100),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        })

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('gbr', gbr)])
        pipeline.fit(X_train, y_train)
        # Perform cross-validation to estimate the model's performance
        # test_size=1/K=20%
        cv = KFold(n_splits=5, random_state=seed_value, shuffle=True)
        mse_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

        # Calculate the mean score and save it
        mean_mse = -mse_scores.mean()

        results_dict = {
            'Tuning_MSE': mean_mse
        }
        csv_filename = f'suni_tuning_mse_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_gradient_boosting_experiment(split, data_augmentation,
                                     feature_selection, algorithm_name,
                                     best_params, num_epochs):
    for i in tqdm(range(num_epochs), desc="Epochs"):
        X_train, y_train, X_test, y_test = Data_loadA2_suni.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Gradient Booster with the best parameter values from tuning
        booster = GradientBoostingRegressor(random_state=seed_value, criterion='squared_error', **best_params)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('booster', booster)])

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Calculate MAPE, RMSE and MAE for the test data
        prediction = pipeline.predict(X_test)
        y_test = np.ravel(y_test)
        prediction = np.ravel(prediction)

        mape, rmse, mae = calculate_metrics(
            y_test, prediction)

        # Create a dictionary to store the results
        results_dict = {
            'learning_rate': best_params['learning_rate'],
            'min_samples_split': best_params['min_samples_split'],
            'min_samples_leaf': best_params['min_samples_leaf'],
            'n_estimators': best_params['n_estimators'],
            'min_weight_fraction_leaf': best_params['min_weight_fraction_leaf'],
            'max_depth': best_params['max_depth'],
            'max_leaf_nodes': best_params['max_leaf_nodes'],
            'max_features': best_params['max_features'],
            'min_impurity_decrease': best_params['min_impurity_decrease'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

        csv_filename = f'suni_params_test_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        # Save prediction results to CSV
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'suni_results_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv',
            index=False)

########################################################################################################################
# Support Vector Machine
# 1. with Feature Selection
# 2. without Feature Selection
########################################################################################################################

def optimize_support_vector_machine_hyperparameters(split, data_augmentation,
                                                    feature_selection, algorithm_name,
                                                    num_trials):
    # Create Optuna study for hyperparameter tuning; maximize the negative error
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_loadA2_suni.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Create a Support Vector Regressor with the suggested hyperparameters
        svr = SVR(kernel='rbf', **{
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
            'C': trial.suggest_float('C', 0.001, 100.0),
            'gamma': trial.suggest_float('gamma', 0.001, 10.0)
        })

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('svr', svr)])

        # Perform cross-validation to estimate the model's performance
        # test_size=1/K=20%
        cv = KFold(n_splits=5, random_state=seed_value, shuffle=True)
        mse_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

        # Calculate the mean score and save it
        mean_mse = -mse_scores.mean()

        results_dict = {
            'Tuning_MSE': mean_mse
        }
        csv_filename = f'suni_tuning_mse_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_support_vector_machine_experiment(split, data_augmentation,
                                          feature_selection, algorithm_name,
                                          best_params, num_epochs):
    for i in tqdm(range(num_epochs), desc="Epochs"):
        X_train, y_train, X_test, y_test = Data_loadA2_suni.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Gradient Booster with the best parameter values from tuning
        svr = SVR(**best_params)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('svr', svr)])

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Calculate MAPE, RMSE and MAE for the test data
        prediction = pipeline.predict(X_test)
        y_test = np.ravel(y_test)
        prediction = np.ravel(prediction)

        mape, rmse, mae = calculate_metrics(
            y_test, prediction)

        # Create a dictionary to store the results
        results_dict = {
            'C': best_params['C'],
            'epsilon': best_params['epsilon'],
            'gamma': best_params['gamma'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }
        csv_filename = f'suni_params_test_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        # Save prediction results to CSV
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'suni_results_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv',
            index=False)
