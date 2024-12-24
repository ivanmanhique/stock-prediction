import pandas as pd
from fastapi import UploadFile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import load_model,Sequential
import os.path as path
import joblib

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

def getStockDataForModel(data : UploadFile) -> pd.DataFrame:
    data = data.file
    df = pd.read_csv(data)
    return df


def prepareDataforModel(df: pd.DataFrame, newModelname):
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    df['MA_50'] = df['Close'].rolling(window=50).mean()

    df['Volatility'] = df['Close'].rolling(window=20).std()

    _data = df.filter(['Close'])
    dataset = _data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    data_folder = path.join(path.dirname(__file__), '../../', 'data')
    scaler_path = path.join(data_folder, f"{newModelname}_scaler.pkl")
    scaled_data = scaler.fit_transform(dataset)
    joblib.dump(scaler, scaler_path)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def continueTrain(model_name:str, train_input:UploadFile, newModelname:str):
    # Define the path to the 'data' folder
    data_folder = path.join(path.dirname(__file__), '../../', 'data')
    model_path = path.join(data_folder, model_name)

    if not path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model( path.join(data_folder, model_name), compile=False)

    df = getStockDataForModel(train_input)

    x_train, y_train = prepareDataforModel(df, newModelname)

    model.compile(
        optimizer='adam',  # Or the optimizer used in the original model
        loss='mean_squared_error',  # Or the original loss function
        metrics=['accuracy']  # Or the original metrics
    )
    # Continue training the model
    history = model.fit(x_train, y_train, batch_size=1, epochs=1)
    save_model_folder= path.join(data_folder,newModelname+".h5" )
    model.save(save_model_folder)

    metrics = history.history

    return metrics


def prepareDataForTest(df: pd.DataFrame):
    x_test = np.array(df)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    return x_test

def predict(model_name: str, test_input:UploadFile):
# Define the path to the 'data' folder
    data_folder = path.join(path.dirname(__file__), '../../', 'data')
    model_path = path.join(data_folder, model_name)
    model_scaler_name = model_name.rsplit('.h5', 1)[0]

    scaler_path = path.join(data_folder, f"{model_scaler_name}_scaler.pkl")

    if not path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

     # Load the scaler
    scaler = joblib.load(scaler_path)
    df = getStockDataForModel(test_input)
    model = load_model(path.join(data_folder, model_name))
    x_test = prepareDataForTest(df)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.tolist()

