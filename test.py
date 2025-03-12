import os
import datetime
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as python_random
import tensorflow as tf
from bs4 import BeautifulSoup
import xml.etree.ElementTree as Xet
from argparse import ArgumentParser
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss
from sklearn.preprocessing import MinMaxScaler
from nbeats_pytorch.model import NBeatsNet

#from bg_forecastcf import BGForecastCF
from forecast import Forecaster

class DataLoader:
    """
    Load data into desired formats for training/validation/testing, including preprocessing.
    """

    def __init__(self, horizon, back_horizon):
        self.horizon = horizon
        self.back_horizon = back_horizon
        self.scaler = list()
        self.historical_values = list()  # first by patient idx, then by col_idx

    def preprocessing(
        self,
        lst_train_arrays,
        lst_test_arrays,
        # train_mode=True, # flag for train_mode (split into train/val), test_mode (no split)
        train_size=0.8,
        normalize=False,
        sequence_stride=6,
        target_col=0,
        horizon=12
    ):
        self.normalize = normalize
        self.sequence_stride = sequence_stride
        self.target_col = target_col
        train_arrays = copy.deepcopy(lst_train_arrays)
        test_arrays = copy.deepcopy(lst_test_arrays)
        # count valid timesteps for each individual series
        # train_array.shape = n_timesteps x n_features
        self.valid_steps_train = [train_array.shape[0] for train_array in train_arrays]
        train_lst, val_lst, test_lst = list(), list(), list()
        for idx in range(len(train_arrays)):
            print(idx, "index")
            bg_sample_train = train_arrays[idx]
            #bg_sample_train_exog = np.delete(train_arrays[idx], 0, 1)
            bg_sample_test = test_arrays[idx]#[:, target_col]
            #bg_sample_test_exog = np.delete(test_arrays[idx], 0, 1)
            valid_steps_sample = self.valid_steps_train[idx]
            #train_target = bg_sample_train_target[: int(train_size * valid_steps_sample)].copy()
            train = bg_sample_train[: int(train_size * valid_steps_sample), :].copy()
            #val_target = bg_sample_train_target[int(train_size * valid_steps_sample) :].copy()
            val = bg_sample_train[int(train_size * valid_steps_sample) :, :].copy()
            #test_target = bg_sample_test_target[:].copy()
            test = bg_sample_test[:, :].copy()
            if self.normalize:
                scaler_cols = list()
                # train.shape = n_train_timesteps x n_features
                for col_idx in range(train.shape[1]):
                    scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
                    train[:, col_idx] = remove_extra_dim(
                        scaler.fit_transform((add_extra_dim(train[:, col_idx])))
                    )
                    val[:, col_idx] = remove_extra_dim(
                        scaler.transform(add_extra_dim(val[:, col_idx]))
                    )
                    test[:, col_idx] = remove_extra_dim(
                        scaler.transform(add_extra_dim(test[:, col_idx]))
                    )
                    scaler_cols.append(scaler)  # by col_idx, each feature
                
                #scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
                #train_target[:] = remove_extra_dim(
                #    scaler.fit_transform((add_extra_dim(train_target[:])))
                #)
                #val_target[:] = remove_extra_dim(
                #    scaler.transform(add_extra_dim(val_target[:]))
                #)
                #test_target[:] = remove_extra_dim(
                #    scaler.transform(add_extra_dim(test_target[:]))
                #)
                #scaler_cols.append(scaler)  # by col_idx, each feature
                self.scaler.append(scaler_cols)  # by pat_idx, each patient
                
            #train = np.column_stack((np.transpose(train_target), train_exog))
            #val = np.column_stack((np.transpose(val_target), val_exog))
            #print(train, val)
            lst_hist_values = list()
            for col_idx in range(train.shape[1]):
                all_train_col = np.concatenate((train[:, col_idx], val[:, col_idx]))
                # decimals = 1, 2 OR 3?
                unique_values = np.unique(np.round(all_train_col, decimals=2))
                lst_hist_values.append(unique_values)
            self.historical_values.append(lst_hist_values)

            train_lst.append(train)
            #train_lst_exog.append(train_exog)
            val_lst.append(val)
            #val_lst_exog.append(val_exog)
            test_lst.append(test)
            #test_lst_exog.append(test_exog)

        (
            self.X_train,
            self.Y_train,
            self.train_idxs,
        ) = self.create_sequences(
            train_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
        (
            self.X_val,
            self.Y_val,
            self.val_idxs,
        ) = self.create_sequences(
            val_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
        (
            self.X_test,
            self.Y_test,
            self.test_idxs,
        ) = self.create_sequences(
            test_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
    @staticmethod
    def create_sequences(
        series_lst, horizon, back_horizon, sequence_stride, target_col=0, exog=False
    ):
        Xs, Ys, sample_idxs = list(), list(), list()
        
        cnt_nans = 0
        for idx, series in enumerate(series_lst):
            len_series = series.shape[0]
            if len_series < (horizon + back_horizon):
                print(
                    f"Warning: not enough timesteps to split for sample {idx}, len: {len_series}, horizon: {horizon}, back: {back_horizon}."
                )
            for i in range(0, len_series - back_horizon - horizon, sequence_stride):
                input_series = series[i : (i + back_horizon)]
                output_series = series[
                    (i + back_horizon) : (i + back_horizon + horizon), [target_col]
                ]
                # TODO: add future plans as additional variables (?)
                if np.isfinite(input_series).all() and np.isfinite(output_series).all():
                    Xs.append(input_series)
                    Ys.append(output_series)
                    # record the sample index when splitting
                    sample_idxs.append(idx)
                else:
                    cnt_nans += 1
                    if cnt_nans % 100 == 0:
                        print(f"{cnt_nans} strides skipped due to NaN values.")
        #print("train", np.array(Xs), "test", np.array(Ys), "val", np.array(sample_idxs))
        return np.array(Xs), np.array(Ys), np.array(sample_idxs)

# remove an extra dimension
def remove_extra_dim(input_array):
    # 2d to 1d
    if len(input_array.shape) == 2:
        return np.reshape(input_array, (-1))
    # 3d to 2d (remove the last empty dim)
    elif len(input_array.shape) == 3:
        return np.squeeze(np.asarray(input_array), axis=-1)
    else:
        print("Not implemented.")
        #print(input_array, "JLNA;iknb")

# add an extra dimension
def add_extra_dim(input_array):
    # 1d to 2d
    if len(input_array.shape) == 1:
        return np.reshape(input_array, (-1, 1))
    # 2d to 3d
    elif len(input_array.shape) == 2:
        return np.asarray(input_array)[:, :, np.newaxis]
    else:
        print("Not implemented.")
        #print(input_array, "ALVNAPNV")

# Method: Fix the random seeds to get consistent models
def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)       
        
def prepare_data(dataset, data_path):
    df = []
    df = pd.DataFrame(df)
    if dataset == "simulated":
        for i,j in zip(["01","02","03","04","05","06","07","08","09","10"],[1,2,3,4,5,6,7,8,9,10]):
            a = pd.read_csv(f"../results/simulation_4/adult#0{i}.csv")
            a["Time"] = a[["Time"]].apply(
                lambda x: pd.to_datetime(x, errors="coerce", format="%Y-%m-%d %H:%M:%S")
            )
            #a['Time'] = pd.to_datetime(a['Time'])
            #a.rename(columns={"Time":"ds", "BG":"y"}, inplace=True)
            a = a.dropna()
            #date_index = pd.date_range(a.Time[0], periods=len(a),freq='3min')
            #a.index = date_index
            a['patient_id'] = pd.Series([f"{j}" for x in range(len(a.index))])
            df = pd.concat([df,a], ignore_index=True)
        
        #df.drop(['Time','BG','LBGI','HBGI','Risk'], axis=1, inplace=True)
        print("aldingvapnb[", df)
        idx = int( df.shape[0] * 0.8)#TEST_SIZE)
        cut = int((df.shape[0]-idx)/10)
        Y_train_df = df[df.CGM<df['CGM'].values[-cut]] # 132 train
        Y_test_df = df[df.CGM>=df['CGM'].values[-cut]].reset_index(drop=True) # 12 test  
        Y_train_df.to_csv("data/data_simulation/all_train.csv")
        Y_test_df.to_csv("data/data_simulation/all_test.csv")
        df.drop(['Time','BG','LBGI','HBGI','Risk'], axis=1, inplace=True)
        #return df
        
    elif dataset == "ohiot1dm":
        train = []
        test = []
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        for i in [540, 544, 552, 567, 584, 596, 559, 563, 570, 575, 588, 591]:
            file_train = pd.read_csv(data_path + "data_OhioT1DM/" + f"{i}_train.csv")
            file_test = pd.read_csv(data_path + "data_OhioT1DM/" + f"{i}_test.csv")
            
            file_train['patient_id'] = pd.Series([f"{i}" for x in range(len(file_train.index))])
            file_test['patient_id'] = pd.Series([f"{i}" for x in range(len(file_test.index))])
            
            train = pd.concat([train, file_train], ignore_index=True)
            test = pd.concat([train, file_test], ignore_index=True)
            
        train.to_csv(data_path + "data_OhioT1DM/all_train.csv")
        test.to_csv(data_path + "data_OhioT1DM/all_test.csv")
        
def load_data(dataset, data_path):
    prepare_data(dataset, data_path)
    if dataset == "ohiot1dm":
        train, orig_train = load_ohio_data(data_path, "all_train.csv")
        test, orig_test = load_ohio_data(data_path, "all_test.csv")
    elif dataset == "simulated":
        #idx = int( df.shape[0] * 1-TEST_SIZE )
        #cut = int((df.shape[0]-idx)/10)
        #train = df[df.CGM<df['CGM'].values[-cut]] # 132 train
        #test = df[df.CGM>=df['CGM'].values[-cut]].reset_index(drop=True) # 12 test  
        train, orig_train = load_sim_data(data_path, "all_train.csv")
        test, orig_test = load_sim_data(data_path, "all_test.csv")
    else:
        print("No dataset chosen")
    return train, test, orig_train, orig_test

def load_ohio_data(data_path, file_name="all_train.csv"):
    # load all the patients, combined
    data = pd.read_csv(data_path + "data_OhioT1DM/" + file_name)

    from functools import reduce
    from operator import or_ as union

    def idx_union(mylist):
        idx = reduce(union, (index for index in mylist))
        return idx

    idx_missing = data.loc[data["missing"] != -1].index
    idx_missing_union = idx_union([idx_missing - 1, idx_missing])

    data = data.drop(idx_missing_union)
    data_bg = data[
        [
            "index_new",
            "patient_id",
            "glucose",
            "basal",
            "bolus",
            "carbs",
            "exercise_intensity",
        ]
    ]
    data_bg["time"] = data_bg[["index_new"]].apply(
        lambda x: pd.to_datetime(x, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    )
    data_bg = data_bg.drop("index_new", axis=1)

    data_bg["bolus"][data_bg["bolus"] == -1] = 0
    data_bg["carbs"][data_bg["carbs"] == -1] = 0
    data_bg["exercise_intensity"][data_bg["exercise_intensity"] == -1] = 0
    data_bg["glucose"][data_bg["glucose"] == -1] = np.NaN

    lst_patient_id = [
        540,
        544,
        552,
        567,
        584,
        596,
        559,
        563,
        570,
        575,
        588,
        591,
    ]
    lst_arrays = list()
    for pat_id in lst_patient_id:
        lst_arrays.append(
            np.asarray(
                data_bg[data_bg["patient_id"] == pat_id][
                    [
                        "glucose",
                        "basal",
                        "bolus",
                        "carbs",
                        "exercise_intensity",
                    ]
                ]
            )
        )
    return lst_arrays, data_bg


def load_sim_data(data_path, file_name="all_train.csv"):
    data = pd.read_csv(data_path + "data_simulation/" + file_name)
    data_bg = data[["patient_id", "Time", "CGM", "CHO", "insulin"]]
    #print(data_bg)
    data_bg["time"] = data_bg[["Time"]].apply(
        lambda x: pd.to_datetime(x, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    )
    data_bg = data_bg.drop("Time", axis=1)
    lst_patient_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lst_arrays = list()
    for pat_id in lst_patient_id:
        lst_arrays.append(
            np.asarray(
                data_bg[data_bg["patient_id"] == pat_id][["CGM", "CHO", "insulin"]]
            )
        )
    return lst_arrays, data_bg
    
def forecast_metrics(dataset, Y_pred, inverse_transform=True):
    Y_test_original, Y_pred_original = list(), list()
    if inverse_transform:
        for i in range(dataset.X_test.shape[0]):
            idx = dataset.test_idxs[i]
            scaler = dataset.scaler[idx]

            Y_test_original.append(
                scaler[dataset.target_col].inverse_transform(dataset.Y_test[i])
            )
            Y_pred_original.append(
                scaler[dataset.target_col].inverse_transform(Y_pred[i])
            )

        Y_test_original = np.array(Y_test_original)
        Y_pred_original = np.array(Y_pred_original)
    else:
        Y_test_original = dataset.Y_test
        Y_pred_original = Y_pred

    def smape(Y_test, Y_pred):
        # src: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
        def smape_sample(actual, forecast):
            return 200 * np.mean(
                np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))
            )

        return np.mean([smape_sample(Y_test[i], Y_pred[i]) for i in range(len(Y_pred))])

    def rmse(Y_test, Y_pred):
        return np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    mean_smape = smape(Y_test_original, Y_pred_original)
    mean_rmse = rmse(Y_test_original, Y_pred_original)

    return mean_smape, mean_rmse

def polynomial_values(shift, change_percent, poly_order, horizon, desired_steps=None):
    """
    shift: e.g., +0.1 (110% of the start value)
    change_percent: e.g., 0.1 (10% increase)
    poly_order: e.g., order 1, or 2, ...
    horizon: the forecasting horizon
    desired_steps: the desired timesteps for the change_percent to finally happen (can be larger than horizon)
    """
    if horizon == 1:
        return np.asarray([shift + change_percent])
    desired_steps = desired_steps if desired_steps else horizon

    p_orders = [shift]  # intercept
    p_orders.extend([0 for i in range(poly_order)])
    p_orders[-1] = change_percent / ((desired_steps - 1) ** poly_order)

    p = np.polynomial.Polynomial(p_orders)
    p_coefs = list(reversed(p.coef))
    value_lst = np.asarray([np.polyval(p_coefs, i) for i in range(desired_steps)])

    return value_lst[:horizon]


def generate_bounds(
    center,
    shift,
    desired_center,
    poly_order,
    horizon,
    fraction_std,
    input_series,
    desired_steps,
):
    if center == "last":
        start_value = input_series[-1]
    elif center == "median":
        start_value = np.median(input_series)
    elif center == "mean":
        start_value = np.mean(input_series)
    elif center == "min":
        start_value = np.min(input_series)
    elif center == "max":
        start_value = np.max(input_series)
    else:
        print("Center: not implemented.")

    std = np.std(input_series)
    # Calculate the change_percent based on the desired center (in 2 hours)
    change_percent = (desired_center - start_value) / start_value
    # Create a default fluctuating range for the upper and lower bound if std is too small
    fluct_range = fraction_std * std if fraction_std * std >= 0.025 else 0.025
    upper = add_extra_dim(
        start_value
        * (
            1
            + polynomial_values(
                shift, change_percent, poly_order, horizon, desired_steps
            )
            + fluct_range
        )
    )
    lower = add_extra_dim(
        start_value
        * (
            1
            + polynomial_values(
                shift, change_percent, poly_order, horizon, desired_steps
            )
            - fluct_range
        )
    )

    return upper, lower
    
def plot(orig_train, orig_test, Y_preds):
    print(orig_train)
    plt.plot(orig_train[orig_train['patient_id']==540].iloc[-120:]['time'], orig_train[orig_train['patient_id']==540].iloc[-120:]["glucose"], c='black', label='train')
    time_change = datetime.timedelta(minutes=120) 
    #plt.plot(orig_test['time'], orig_test['CGM'], c='blue', label='test')
    plt.plot(orig_train[orig_train['patient_id']==540].iloc[-24:]['time']+time_change, Y_preds, c='red', label='pred')
    plt.legend()
    plt.grid()
    plt.plot()
    plt.show()

def main():
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, help="Choose dataset.")
    parser.add_argument( "--horizon", type=int, help="Horizon of forecasting task.")
    parser.add_argument( "--back-horizon", type=int, help="Back horizon of forecasting task.")
    parser.add_argument( "--random-seed", type=int, default=39, help="Random seed parameter, default 39.")
    parser.add_argument( "--train-size", type=float, default=0.2, help="Proportional size of the training set.")
    parser.add_argument( "--test-group", type=str, default=None, help="Extract random 100 samples from test group, i.e., 'hyper'/'hypo'; default None.")
    args = parser.parse_args()
    data_path = "./data/"
    lst_arrays, lst_arrays_test, orig_train, orig_test = load_data(args.dataset, data_path) #misschien toch load_data gebruiken?
    print(f"The shape of loaded train: {len(lst_arrays)}*{lst_arrays[0].shape}")
    print(f"The shape of test: {len(lst_arrays_test)}*{lst_arrays_test[0].shape}")
    
    print(f"===========Desired trend parameters=============")
    center = "last"
    desired_shift, poly_order = 0, 1
    fraction_std = 1#args.fraction_std
    print(f"center: {center}, desired_shift: {desired_shift};")
    print(f"fraction_std:{fraction_std};")
    print(f"desired_change:'sample_based', poly_order:{poly_order}.")

    
    TARGET_COL = 0
    if args.dataset == "ohiot1dm":
        CHANGE_COLS = [1, 2, 3, 4]
    elif args.dataset == "simulated":
        CHANGE_COLS = [1, 2]
    else:
        CHANGE_COLS = None
    RANDOM_STATE = args.random_seed
    TRAIN_SIZE = args.train_size
    horizon, back_horizon = args.horizon, args.back_horizon
    dataset = DataLoader(horizon, back_horizon)
    dataset.preprocessing(#???
        lst_train_arrays=lst_arrays,
        lst_test_arrays=lst_arrays_test,
        train_size=TRAIN_SIZE,
        normalize=True,
        sequence_stride=horizon,
        target_col=TARGET_COL,
        horizon = args.horizon
    )

    print(dataset.X_train.shape, dataset.Y_train.shape)
    print(dataset.X_val.shape, dataset.Y_val.shape)
    print(dataset.X_test.shape, dataset.Y_test.shape)
    for model_name in ["gru", "seq2seq"]:#, "nbeats", "wavenet"
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        reset_seeds(RANDOM_STATE)
        n_in_features = dataset.X_train.shape[2]
        n_out_features = 1

        ###############################################
        # ## 2.0 Forecasting model
        ###############################################
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        reset_seeds(RANDOM_STATE)
        if model_name in ["wavenet", "seq2seq"]:
            forecast_model = build_tfts_model(
                model_name, back_horizon, horizon, n_in_features
            )
            '''
            elif model_name == "nbeats":
                forecast_model = NBeatsNet(
                    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
                    forecast_length=horizon,
                    backcast_length=back_horizon,
                    hidden_layer_units=256,
                )
                #forecast_model = NBEATSx(
                #    h=horizon,
                #    input_size=2*horizon,
                #    stack_types=['trend'],#(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
                    #backcast_length=back_horizon,
                #    mlp_units=[[512, 512], [512, 512], [512, 512]],
                #    loss=MAE(),
                #)

                # Definition of the objective function and the optimizer
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
                forecast_model.compile(optimizer=optimizer, loss="mae")
            '''
        elif model_name == "gru":
            forecast_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(back_horizon, n_in_features)),
                    # Shape [batch, time, features] => [batch, time, gru_units]
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=True),
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=False),
                    # Shape => [batch, time, features]
                    tf.keras.layers.Dense(horizon, activation="linear"),
                    tf.keras.layers.Reshape((horizon, n_out_features)),
                ]
            )

            # Definition of the objective function and the optimizer
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
            forecast_model.compile(optimizer=optimizer, loss="mae")
        else:
            print("Not implemented: model_name.")

        # Define the early stopping criteria
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
        )

        print(dataset.X_train.shape, dataset.Y_train.shape)
        print(dataset.X_val.shape, dataset.Y_val.shape)
        print(dataset.X_test.shape, dataset.Y_test.shape)
        # Train the model
        reset_seeds(RANDOM_STATE)
        forecast_model.fit(
            dataset.X_train,
            dataset.Y_train,
            epochs=200,
            batch_size=64,
            validation_data=(dataset.X_val, dataset.Y_val),
            #callbacks=[early_stopping],
        )

        # Predict on the testing set (forecast)
        Y_preds = forecast_model.predict(dataset.X_test)
        mean_smape, mean_rmse = forecast_metrics(dataset, Y_preds)
        print(
            f"[[{model_name}]] model trained, with test sMAPE score {mean_smape:0.4f}; test RMSE score: {mean_rmse:0.4f}."
        )
        
        #print(lst_arrays_test)
        #prediction = forecast_model.predict(lst_arrays_test)
        print(dataset.X_train.shape, dataset.Y_train.shape)
        print(dataset.X_val.shape, dataset.Y_val.shape)
        print(dataset.X_test.shape, dataset.Y_test.shape)
        print("PREDOBCOSB", Y_preds)
        print(":AOKEGv[jainb[", dataset.X_train, dataset.Y_train)
        
        hyper_bound, hypo_bound = 180, 70
        print(f"===========CF generation setup=============")
        print(f"hyper bound value: {hyper_bound}, hypo bound: {hypo_bound}.")

        event_labels = list()
        for i in range(len(Y_preds)):
            scaler = dataset.scaler[dataset.test_idxs[i]][TARGET_COL]
            Y_preds_original = scaler.inverse_transform(Y_preds[i])
            if np.any(Y_preds_original >= hyper_bound):
                event_labels.append("hyper")
            elif np.any(Y_preds_original <= hypo_bound):
                event_labels.append("hypo")
            else:
                event_labels.append("normal")
        hyper_indices = np.argwhere(np.array(event_labels) == "hyper").reshape(-1)
        hypo_indices = np.argwhere(np.array(event_labels) == "hypo").reshape(-1)

        print(f"hyper_indices shape: {hyper_indices.shape}")
        print(f"hypo_indices shape: {hypo_indices.shape}")
        
        print("LSASLSLKDGNS", Y_preds_original)
        #plot(orig_train, orig_test, Y_preds_original)
        
        # use a subset of the test
        rand_test_size = 100
        print(args.test_group)
        if args.test_group == "hyper":
            if len(hyper_indices) >= rand_test_size:
                print("if", hyper_indices)
                np.random.seed(RANDOM_STATE)
                rand_test_idx = np.random.choice(
                    hyper_indices, rand_test_size, replace=False
                )
            else:
                print("else", hyper_indices)
                rand_test_idx = hyper_indices
        elif args.test_group == "hypo":
            if len(hypo_indices) >= rand_test_size:
                np.random.seed(RANDOM_STATE)
                rand_test_idx = np.random.choice(
                    hypo_indices, rand_test_size, replace=False
                )
            else:
                rand_test_idx = hypo_indices
        else:
            rand_test_idx = np.arange(dataset.X_test.shape[0])

        X_test = dataset.X_test[rand_test_idx]
        Y_test = dataset.Y_test[rand_test_idx]
        
        print(
            f"Generating CFs for {len(rand_test_idx)} samples in total, for {args.test_group} test group..."
        )

        # loss calculation ==> min/max bounds
        desired_max_lst, desired_min_lst = list(), list()
        hist_inputs = list()

        # define the desired center to reach in two hours (24 timesteps for OhioT1DM)
        # then we need to cut the first 6 steps to generate the desired bounds
        desired_steps = 24 if args.dataset == "ohiot1dm" else 20
        if args.test_group == "hyper":
            desired_center_2h = hyper_bound - 10  # -10 for a fluctuating bound
        elif args.test_group == "hypo":
            desired_center_2h = hypo_bound + 10  # +10 for a fluctuating bound
        else:
            print(
                f"Group not identified: {args.test_group}, use a default center"
            )
            desired_center_2h = (hyper_bound + hypo_bound) / 2
        print(f"desired center {desired_center_2h} in {desired_steps} timesteps.")

        for i in range(len(X_test)):
            idx = dataset.test_idxs[rand_test_idx[i]]
            scaler = dataset.scaler[idx]

            desired_center_scaled = scaler[TARGET_COL].transform(
                np.array(desired_center_2h).reshape(-1, 1)
            )[0][0]
            print(
                f"desired_center: {desired_center_2h}; after scaling: {desired_center_scaled:0.4f}"
            )

            # desired trend bounds: use the `center` parameter from the input sequence as the starting point
            desired_max_scaled, desired_min_scaled = generate_bounds(
                center=center,  # Use the parameters defined at the beginning of the script
                shift=desired_shift,
                desired_center=desired_center_scaled,
                poly_order=poly_order,
                horizon=horizon,
                fraction_std=fraction_std,
                input_series=X_test[i, :, TARGET_COL],
                desired_steps=desired_steps,
            )
            # TODO: remove the ones that already satisfy the bounds here, OR afterwards?
            desired_max_lst.append(desired_max_scaled)
            desired_min_lst.append(desired_min_scaled)
            hist_inputs.append(dataset.historical_values[idx])
            print("AP:IGDJVN{AO", hist_inputs)
        
        # create a dict for step_weights, prediction margin, clip_mechanism, and hist_input
        cf_model = Forecaster(#BGForecastCF(
            max_iter=100,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            #pred_margin_weight=0.9,  # focus 0.9 the prediction bound calculation (then `1-pred_margin_weight` on the input weighted steps)
            step_weights="unconstrained",
            #random_state=RANDOM_STATE,
            target_col=TARGET_COL,
            #only_change_idx=CHANGE_COLS,
            horizon=horizon,
        )
        
if __name__ == "__main__":
    main()
