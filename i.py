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
Scaler_data = Scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train, y_train = [], []

for x in range(prediction_days, len(Scaler_data)):
    x_train.append(Scaler_data[x-prediction_days:x, 0])
    y_train.append(Scaler_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_tain.shape[0], x_train.shape[1], 1))

# Creat Neural Network

model = Sequential()

model.add(LSTM(unit=50, return_sequences=True, input_shape=(x_train.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.Dense(units=1)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Testing The Model

test_start = DT.datetime(2020,1,1)
test_end = datetime.now()

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_strat, test_end)
actual_prices = test_data['Close'].values.reshape

total_dataset = pd.contact((data['Close'], test_data['Close']), axis=0)

model_input = total_dataset[LEN(total_dataset)- len(test_data)- prediction_days:].values
model_input = model_inputs.reshape(-1, 1)
model_input = scaler.fit_transform(model_input)

x_test = []
