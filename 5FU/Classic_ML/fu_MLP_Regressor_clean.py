########################################################################################################################
# MLPRegressor for 5FU project                                                                                         #
# Based on StackOverflow code by Flavia Giammarino                                                                     #
# last edited: 20.02.24  (Changed Python version for MissForest, which currently requires 3.10)                        #
# this has been written in Python 3.10 and may not work the same in future versions!                                   #
#                                                                                                                      #
# data used:                                                                                                           #
# '5fu_data_split_0_aug_preproc.csv': preprocessed data without augmentation  preprocessed data with set               #
# general note: here, dynamic batch sizes are used                                                                     #
########################################################################################################################

import csv
from optuna.samplers import TPESampler
import keras
from keras.regularizers import l1
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm
import Data_load_clean_5fu
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
import optuna
from sklearn.model_selection import KFold
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

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)

# 5. Set the keras seed
keras.utils.set_random_seed(seed_value)


# Metrics

def mape_array(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true)
    return np.mean(absolute_percentage_errors)


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


def calculate_metrics(y_true, y_pred):
    mape = mape_array(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return mape, rmse, mae


def save_results_to_csv(results_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = results_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write the header
        writer.writeheader()
        # Write the data
        writer.writerow(results_dict)


########################################################################################################################
# Neural Network with 2 hidden layers                                                                                  #
########################################################################################################################

def create_model(X_train, first_layer_neurons, second_layer_neurons, activation, learning_rate_init, drop_out,
                 l1_reg):
    model = tf.keras.Sequential()

    # Add input and first hidden layer (combined in one in Keras)
    model.add(tf.keras.layers.Dense(first_layer_neurons, activation=activation, kernel_regularizer=l1(l1_reg),
                                    input_shape=(X_train.shape[1],)))  # input shape set for 2DNumPy array

    model.add(tf.keras.layers.Dropout(drop_out))

    # Add second hidden layer
    model.add(tf.keras.layers.Dense(second_layer_neurons, activation=activation, kernel_regularizer=l1(l1_reg)))

    # Add output layer
    model.add(tf.keras.layers.Dense(1, activation='softplus'))

    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0, learning_rate=learning_rate_init)
    model.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.MeanAbsoluteError(),
                                                            keras.metrics.MeanAbsolutePercentageError()])

    return model  # Return the Keras model


########################################################################################################################
# Neural Network with 1 hidden layer                                                                                   #
########################################################################################################################

def create_model2(X_train, first_layer_neurons, activation, learning_rate_init, drop_out,
                  l1_reg):
    model2 = tf.keras.Sequential()

    # Add input and first hidden layer (combined in one in Keras)
    model2.add(tf.keras.layers.Dense(first_layer_neurons, activation=activation, kernel_regularizer=l1(l1_reg),
                                     input_shape=(X_train.shape[1],)))  # input shape set for 2DNumPy array

    model2.add(tf.keras.layers.Dropout(drop_out))

    # Add output layer
    model2.add(tf.keras.layers.Dense(1, activation='softplus'))

    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0, learning_rate=learning_rate_init)
    model2.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.MeanAbsoluteError(),
                                                             keras.metrics.MeanAbsolutePercentageError()])

    return model2  # Return the Keras model


########################################################################################################################
# Multilayer Perceptron with Two Hidden Layers
########################################################################################################################

def optimize_MLP_two_hidden_layers_hyperparameters(split, data_augmentation,
                                                   algorithm_name,
                                                   num_trials):
    # Create Optuna study for hyperparameter tuning; minimize positive error
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_load_clean_5fu.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            algorithm_name=algorithm_name
        )

        # Initialize K-Fold cross-validation
        n_splits = 5  # Number of folds
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

        mse_scores = []

        # Perform cross-validation
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = create_model(X_train_fold,
                                 first_layer_neurons=trial.suggest_int('first_layer_neurons', 2, 10),
                                 second_layer_neurons=trial.suggest_int('second_layer_neurons', 2, 10),
                                 activation=trial.suggest_categorical('activation',
                                                                      ['elu', 'softplus', 'selu', 'relu']),
                                 learning_rate_init=trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
                                 drop_out=trial.suggest_float('drop_out', 0.05, 0.4),
                                 l1_reg=trial.suggest_float('l1_reg', 1e-5, 1e-1, log=True))

            # wrap the model into KerasRegressor (sklearn) to make it compatible with pipeline
            clf = KerasRegressor(build_fn=model, verbose=0, random_state=seed_value)

            # Scale and Impute data, Create Pipeline
            preprocessor = preprocessing_pipeline(X_train)
            pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])

            # Train the model
            pipeline.fit(X_train_fold, y_train_fold)
            # Make predictions on the validation set
            y_val_pred = pipeline.predict(X_val_fold)

            # Calculate mean squared error and append to scores
            mse = mean_squared_error(y_val_pred, y_val_fold)
            mse_scores.append(mse)

        # Calculate mean of MSE scores
        mean_mse = np.mean(mse_scores)
        results_dict = {
            'Tuning_MSE': mse
        }
        csv_filename = f'5fu_tuning_mse_{algorithm_name}_split_{split}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_mlp_two_hidden_layers_experiment(split, data_augmentation, algorithm_name,
                                         best_params,
                                         num_epochs):
    X_train, y_train, X_test, y_test = Data_load_clean_5fu.train_test_split(
        split=split,
        data_augmentation=data_augmentation,
        algorithm_name=algorithm_name
    )

    for i in tqdm(range(num_epochs), desc="Epochs"):
        model = create_model(X_train,
                             best_params['first_layer_neurons'],
                             best_params['second_layer_neurons'],
                             best_params['activation'],
                             best_params['learning_rate_init'],
                             best_params['drop_out'],
                             best_params['l1_reg'])

        # wrap the model into KerasRegressor (sklearn) to make it compatible with pipeline
        clf = KerasRegressor(build_fn=model, verbose=0, random_state=seed_value)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Calculate MAPE, RMSE and MAE for the test data
        prediction = pipeline.predict(X_test)
        y_test = np.ravel(y_test)
        prediction = np.ravel(prediction)

        mape, rmse, mae = calculate_metrics(
            y_test, prediction)

        results_dict = {
            'first_layer_neurons': best_params['first_layer_neurons'],
            'second_layer_neurons': best_params['second_layer_neurons'],
            'learning_rate': best_params['learning_rate_init'],
            'activation': best_params['activation'],
            'l1_reg': best_params['l1_reg'],
            'drop_out': best_params['drop_out'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

        csv_filename = f'5fu_params_test_{algorithm_name}_split_{split}_augmentation_{data_augmentation}.csv'
        save_results_to_csv(results_dict, csv_filename)

        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'5fu_results_{algorithm_name}_split_{split}_augmentation_{data_augmentation}.csv',
            index=False)


########################################################################################################################
# Multilayer Perceptron with One Hidden Layer
########################################################################################################################

def optimize_MLP_one_hidden_layer_hyperparameters(split, data_augmentation,
                                                  algorithm_name,
                                                  num_trials):
    # Create Optuna study for hyperparameter tuning; minimize positive error
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed_value))

    # Define the objective function
    def objective(trial):
        X_train, y_train, X_test, y_test = Data_load_clean_5fu.train_test_split(
            split=split,
            data_augmentation=data_augmentation,
            algorithm_name=algorithm_name
        )

        # Initialize K-Fold cross-validation
        n_splits = 5  # Number of folds
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

        mse_scores = []

        # Perform cross-validation
        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model2 = create_model2(X_train_fold,
                                   first_layer_neurons=trial.suggest_int('first_layer_neurons', 2, 10),
                                   activation=trial.suggest_categorical('activation',
                                                                        ['elu', 'softplus', 'selu', 'relu']),
                                   learning_rate_init=trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
                                   drop_out=trial.suggest_float('drop_out', 0.05, 0.4),
                                   l1_reg=trial.suggest_float('l1_reg', 1e-5, 1e-1, log=True))

            # wrap the model into KerasRegressor (sklearn) to make it compatible with pipeline
            clf = KerasRegressor(build_fn=model2, verbose=0, random_state=seed_value)

            # Scale and Impute data, Create Pipeline
            preprocessor = preprocessing_pipeline(X_train)
            pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])

            # Train the model
            pipeline.fit(X_train_fold, y_train_fold)

            # Make predictions on the validation set
            y_val_pred = pipeline.predict(X_val_fold)

            # Calculate mean squared error and append to scores
            mse = mean_squared_error(y_val_pred, y_val_fold)
            mse_scores.append(mse)

        # Calculate mean of MSE scores
        mean_mse = np.mean(mse_scores)
        results_dict = {
            'Tuning_MSE': mse
        }
        csv_filename = f'5fu_tuning_mse_{algorithm_name}_split_{split}_augmentation_{data_augmentation}.csv'

        # Save parameters to the CSV file
        save_results_to_csv(results_dict, csv_filename)

        return mean_mse

    # Start the optimization process
    study.optimize(objective, n_trials=num_trials)

    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    return best_params


def run_mlp_one_hidden_layer_experiment(split, data_augmentation, algorithm_name,
                                        best_params,
                                        num_epochs):
    X_train, y_train, X_test, y_test = Data_load_clean_5fu.train_test_split(
        split=split,
        data_augmentation=data_augmentation,
        algorithm_name=algorithm_name
    )

    for i in tqdm(range(num_epochs), desc="Epochs"):

        model2 = create_model2(X_train,
                               best_params['first_layer_neurons'],
                               best_params['activation'],
                               best_params['learning_rate_init'],
                               best_params['drop_out'],
                               best_params['l1_reg'])

        # wrap the model into KerasRegressor (sklearn) to make it compatible with pipeline
        clf = KerasRegressor(build_fn=model2, verbose=0, random_state=seed_value)

        # Scale and Impute data, Create Pipeline
        preprocessor = preprocessing_pipeline(X_train)
        pipeline = Pipeline([('preprocessor', preprocessor), ('clf', clf)])

        # Train the model on the training data
        pipeline.fit(X_train, y_train)

        # Calculate MAPE, RMSE and MAE for the test data
        prediction = pipeline.predict(X_test)
        y_test = np.ravel(y_test)
        prediction = np.ravel(prediction)

        mape, rmse, mae = calculate_metrics(
            y_test, prediction)

        results_dict = {
            'first_layer_neurons': best_params['first_layer_neurons'],
            'learning_rate': best_params['learning_rate_init'],
            'activation': best_params['activation'],
            'l1_reg': best_params['l1_reg'],
            'drop_out': best_params['drop_out'],
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae
        }

        csv_filename = f'5fu_params_test_{algorithm_name}_split_{split}_augmentation_{data_augmentation}.csv'
        save_results_to_csv(results_dict, csv_filename)

        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
        results_df.to_csv(
            f'5fu_results_{algorithm_name}_split_{split}_augmentation_{data_augmentation}.csv',
            index=False)
