import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pandas_datareader as web
import datetime as dt   

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()


data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)


# Prepare Data
Scaler = MinMaxScaler(feature_range=(0,1))
Scaler_data = Scaler.fit_transform(data['Close'].values.reshape(-,1,1))

prediction_days = 60

x_train, y_train = [], []

for x in range(prediction_days, len(Scaler_data)):
    x_train.append(Scaler_data[x-prediction_days:x, 0])
    y_train.append(Scaler_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_tain.shape[0], x_train.shape[1], 1))