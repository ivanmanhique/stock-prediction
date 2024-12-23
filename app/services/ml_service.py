import pandas as pd
from fastapi import UploadFile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import load_model
import os.path as path



def getStockDataForModelTraining(data : UploadFile) -> pd.DataFrame:
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
    data_folder = path.join(path.dirname(__file__), '..', 'data')


    model = load_model( path.join(data_folder, model_name))
    df = getStockDataForModelTraining(train_input)

    x_train, y_train = prepareDataforModelTraining(df)

    # Continue training the model
    history = model.fit(x_train, y_train, batch_size=1, epochs=1)

    model.save(newModelname)

    metrics = history.history

    return metrics
