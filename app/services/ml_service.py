import pandas as pd
from fastapi import UploadFile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import load_model
import os.path as path

import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

def getStockDataForModelTraining(data : UploadFile) -> pd.DataFrame:
    data = data.file
    df = pd.read_csv(data)
    return df


def prepareDataforModelTraining(df: pd.DataFrame):
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    df['MA_50'] = df['Close'].rolling(window=50).mean()

    df['Volatility'] = df['Close'].rolling(window=20).std()

    _data = df.filter(['Close'])
    dataset = _data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

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

    print("Resolved data folder:", data_folder)
    print("Resolved model path:", path.join(data_folder, model_name))
    model_path = path.join(data_folder, model_name)

    if not path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")


    model = load_model( path.join(data_folder, model_name), compile=False)

    df = getStockDataForModelTraining(train_input)

    x_train, y_train = prepareDataforModelTraining(df)

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