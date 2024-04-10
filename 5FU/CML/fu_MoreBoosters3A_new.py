########################################################################################################################
# XGBoost, LightGBM for 5FU Project                                                                                    #
# Based on OtherMethods_fin                                                                                            #
# last edited: 20.02.24  (Changed Python version for MissForest, which currently requires 3.10)                        #
# this has been written in Python 3.10 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# '5fu_data_preproc_aug0_split.csv': preprocessed data without augmentation  preprocessed data with set                #
#                                                                                                                      #
# general note: cross_validate and cross_val_score need some negative metrics (like neg. MSE)                          #
# because they expect that higher values are better. However, we want to report positive metrics; therefore            #
# before saving the metric, we need a (-) sign.                                                                        #
########################################################################################################################

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import Data_loadA2_5fu
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
    # Define the pipeline: Scaling with MinMaxScaler, One-hot encoding with
    # OneHotEncoder

    scaler, onehot = MinMaxScaler(), OneHotEncoder()

    # Select columns
    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = make_pipeline(scaler)
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
# XGBoost
# 1. with Feature Selection
# 2. without Feature Selection
########################################################################################################################

def optimize_xtreme_gradient_boosting_hyperparameters(split, data_augmentation,
                                                      feature_selection, algorithm_name,
                                                      num_trials):
    # Create Optuna studies for hyperparameter tuning; maximize negative error
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_loadA2_5fu.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )
        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Create a Xtreme Gradient Booster with the suggested hyperparameters
        xgb = XGBRegressor(seed=seed_value, objective='reg:squarederror', **{
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
        })

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('xgb', xgb)])

        # Perform cross-validation to estimate the model's performance
        # test_size=1/K=20%
        cv = KFold(n_splits=5, random_state=seed_value, shuffle=True)
        mse_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

        # Calculate the mean score
        mean_mse = -mse_scores.mean()

        results_dict = {
            'Tuning_MSE': mean_mse
        }
        csv_filename = f'5fu_tuning_mse_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_xtreme_gradient_boosting_experiment(split, data_augmentation,
                                            feature_selection, algorithm_name,
                                            best_params, num_epochs):
    for i in tqdm(range(num_epochs), desc="Epochs"):
        X_train, y_train, X_test, y_test = Data_loadA2_5fu.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Xtreme Gradient Booster with the best parameter values from tuning
        xgbooster = XGBRegressor(seed=seed_value, objective='reg:squarederror', **best_params)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('xgbooster', xgbooster)])

        # Fit the pipeline on the training data
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
            'n_estimators': best_params['n_estimators'],
            'max_depth': best_params['max_depth'],
            'min_child_weight': best_params['min_child_weight'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'gamma': best_params['gamma'],
            'reg_lambda': best_params['reg_lambda'],
            'reg_alpha': best_params['reg_alpha'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }
        csv_filename = f'5fu_params_test_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        # Save prediction results to CSV
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'5fu_results_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv',
            index=False)


########################################################################################################################
# LightGBM
# 1. with Feature Selection
# 2. without Feature Selection
########################################################################################################################

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, message="[LightGBM]")


def optimize_light_gradient_boosting_hyperparameters(split, data_augmentation,
                                                     feature_selection, algorithm_name,
                                                     num_trials):
    # Create Optuna study for hyperparameter tuning; maximize negative error
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_loadA2_5fu.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Create a Light Gradient Booster with the suggested hyperparameters
        lgb = LGBMRegressor(random_state=seed_value, objective='mse', verbose=-1, **{
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0)
        })

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('lgb', lgb)])

        # Perform cross-validation to estimate the model's performance
        # test_size=1/K=20%
        cv = KFold(n_splits=5, random_state=seed_value, shuffle=True)
        mse_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

        # Calculate the mean score
        mean_mse = -mse_scores.mean()

        results_dict = {
            'Tuning_MSE': mean_mse
        }
        csv_filename = f'5fu_tuning_mse_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_light_gradient_boosting_experiment(split, data_augmentation,
                                           feature_selection, algorithm_name,
                                           best_params, num_epochs):
    for i in tqdm(range(num_epochs), desc="Epochs"):
        X_train, y_train, X_test, y_test = Data_loadA2_5fu.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            feature_selection=feature_selection,
            algorithm_name=algorithm_name
        )

        # ravel() to convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)

        # Light Gradient Booster with the best parameter values from tuning
        lgbooster = LGBMRegressor(random_state=seed_value, objective='mse', verbose=-1, **best_params)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('lgbooster', lgbooster)])

        # Fit the pipeline on the training data
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
            'n_estimators': best_params['n_estimators'],
            'max_depth': best_params['max_depth'],
            'num_leaves': best_params['num_leaves'],
            'min_child_samples': best_params['min_child_samples'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'min_split_gain': best_params['min_split_gain'],
            'min_child_weight': best_params['min_child_weight'],
            'reg_lambda': best_params['reg_lambda'],
            'reg_alpha': best_params['reg_alpha'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

        csv_filename = f'5fu_params_test_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        # Save prediction results to CSV
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'5fu_results_{algorithm_name}_split_{split}_{"FS" if feature_selection else "No_FS"}_augmentation_{data_augmentation}.csv',
            index=False)
