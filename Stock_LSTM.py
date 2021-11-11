import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def predict_stocks(data, stock, days, hidden_layers, nodes, training_epochs):
    stock_data = data[data['ticker_symbol'] == stock].copy()
    training_data_0 = stock_data[stock_data['day_date'] < '2019-01-01'].copy()
    testing_data = stock_data[stock_data['day_date'] >= '2019-01-01'].copy()
    
    training_data = training_data_0.drop(['ticker_symbol', 'day_date'], axis=1)
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    y_train = []

    for i in range(days, training_data.shape[0]):
        X_train.append(training_data[i-days:i])
        y_train.append(training_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    regression = Sequential()

    regression.add(LSTM(units = days, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
    regression.add(Dropout(0.2))

    for i in range(hidden_layers):
        if (i == hidden_layers - 1):
            regression.add(LSTM(units = nodes[i], activation = 'relu'))
        else:
            regression.add(LSTM(units = nodes[i], activation = 'relu', return_sequences = True))
        regression.add(Dropout(0.2))

    regression.add(Dense(units = 1))

    regression.compile(optimizer='adam', loss='mean_squared_error')
    regression.fit(X_train, y_train, epochs=training_epochs, batch_size=32)

    past_days = training_data_0.tail(days)
    inputs = past_days.append(testing_data, ignore_index=True)
    inputs = inputs.drop(['ticker_symbol', 'day_date'], axis=1)
    inputs = scaler.transform(inputs)

    X_test = []
    y_test = []

    for i in range(days, inputs.shape[0]):
        X_test.append(inputs[i-days:i])
        y_test.append(inputs[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = regression.predict(X_test)
    scale = 1/scaler.scale_[0]
    y_pred = y_pred*scale
    y_test = y_test*scale

    plt.figure(figsize=(14,5))
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(y_pred, color='blue', label='Predicted Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    rmse = np.sqrt((1/y_pred.size) * np.sum(np.square(y_pred - y_test)))
    print(rmse)
    return rmse

def predict_percent_changes(data, stock, days, hidden_layers, nodes, dropout, training_epochs):
    stock_data = data[data['ticker_symbol'] == stock].copy()
    percent_changes = []
    for i in range(stock_data.shape[0] - 1):
        percent_changes.append((stock_data.iat[i+1,2] - stock_data.iat[i,2])/stock_data.iat[i,2])
    stock_data = stock_data[:-1]
    stock_data.insert(2, 'percent_changes', percent_changes)
    training_data_0 = stock_data[stock_data['day_date'] < '2019-01-01'].copy()
    testing_data = stock_data[stock_data['day_date'] >= '2019-01-01'].copy()
    
    training_data = training_data_0.drop(['ticker_symbol', 'day_date'], axis=1)
    scaler = MinMaxScaler(feature_range = (-1,1))
    training_data = scaler.fit_transform(training_data)

    X_train = []
    y_train = []

    for i in range(days, training_data.shape[0]):
        X_train.append(training_data[i-days:i])
        y_train.append(training_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    regression = Sequential()

    regression.add(LSTM(units = days, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 6)))
    regression.add(Dropout(0.2))

    for i in range(hidden_layers):
        if (i == hidden_layers - 1):
            regression.add(LSTM(units = nodes[i], activation = 'relu'))
        else:
            regression.add(LSTM(units = nodes[i], activation = 'relu', return_sequences = True))
        regression.add(Dropout(dropout))

    regression.add(Dense(units = 1))

    regression.compile(optimizer='adam', loss='mean_squared_error')
    regression.fit(X_train, y_train, epochs=training_epochs, batch_size=32)

    past_days = training_data_0.tail(days)
    inputs = past_days.append(testing_data, ignore_index=True)
    inputs = inputs.drop(['ticker_symbol', 'day_date'], axis=1)
    inputs = scaler.transform(inputs)

    X_test = []
    y_test = []

    for i in range(days, inputs.shape[0]):
        X_test.append(inputs[i-days:i])
        y_test.append(inputs[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    y_pred = regression.predict(X_test)
    scale = 1/scaler.scale_[0]
    y_pred = y_pred*scale
    y_test = y_test*scale

    plt.figure(figsize=(14,5))
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(y_pred, color='blue', label='Predicted Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    rmse = np.sqrt((1/y_pred.size) * np.sum(np.square(y_pred - y_test)))
    print(rmse)
    print(y_pred)
    return rmse
