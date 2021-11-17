import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import itertools
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


"""
To-Do:
- VaderSentiment
    1. First note, we don't HAVE to implement this for the Midpoint part
    2. We just need to use Vader to add the average sentiment of tweets about a company from that day as a feature
    3. Perhaps we should weigh sentiment by retweets or likes, since this data is given to us
- Feature Selection or Creation. 
    1. Adding or Taking away features from the stock data to see what gives the best model
    2. Engienering new features from financial indicators, e.g. averages of stock prices from the past x number of days
       (If the new features weren't time series data, then we should use a different predictor that isn't LSTM)
- Cross-validation
    1. Need to implement a method to do cross-validation on the hyperparamters of the models that we are using for our predictions
- New methods of prediction
    1. Just a simple linear regression between stock prices or different features
    2. A regression between average sentiment of tweets from a certain day
    3. Or any other possible type of neural net or regression we could use for prediction purposes
"""


def kfold(data, stock, kf):

    rmse_per_days_per_features = []

    feat = [0, 1, 2, 3]
    for f in range(0, len(feat)+1):
        for sub_f in itertools.combinations(feat, f):
            sub_f = list(sub_f)
            print(sub_f)

            d = 5

            # i ranges from 1 to 8
            for i in range(1, 9):

                # days ranges from 5 to 40
                days = d * i

                X_train, y_train, X_test, y_test, scale = preprocessing(data, stock, days, features=sub_f)
                kfold = model_selection.KFold(n_splits=kf)
                avg_rmse = 0
                for train, test in kfold.split(X_train, y_train):
                    rmse = predict_stocks_LSTM(X_train[train], y_train[train], X_train[test], y_train[test], scale, days, 1, [days, days], 0.2, 5, False)
                    avg_rmse += rmse
                avg_rmse /= kf
                rmse_per_days_per_features.append([days, avg_rmse, sub_f])

    print(" ")
    print(" ")
    for day_rmse in rmse_per_days_per_features:
        print("days = ", day_rmse[0], ", avg_rmse = ", day_rmse[1], ", features = ", day_rmse[2])


def best_combinations(data, stock, kf, days, features):

    X_train, y_train, X_test, y_test, scale = preprocessing(data, stock, days, features)
    rmse = predict_stocks_LSTM(X_train, y_train, X_test, y_test, scale, days, 1, [days], 0.2, 5, True)
    return rmse



"""
Method to split data into features/labels and training/testing
Arguments:
- data: The Pandas Dataframe holding the stock data
- stock: String with ticker_symbol of which stock you want to look at
- days: How many days of data are going to be used to predict a stock price
- features: List of integers detailing which features are going to be kept
"""


def preprocessing(data, stock, days, features=[0,1,2,3]):
    #Get stock data for specific stock
    stock_data = data[data['ticker_symbol'] == stock].copy()

    #Drops features not in feature list 
    dropped_features = []
    for i in range(4):
        if i not in features:
            dropped_features.append(i+3)
    stock_data.drop(columns = stock_data.columns[dropped_features], axis=1, inplace=True)

    #Split data into testing and training, using 8 years and 2 years from Train/Test Split
    training_data_0 = stock_data[stock_data['day_date'] < '2019-01-01'].copy()
    testing_data = stock_data[stock_data['day_date'] >= '2019-01-01'].copy()
    
    training_data = training_data_0.drop(['ticker_symbol', 'day_date'], axis=1)
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    scale = 1/scaler.scale_[0]

    X_train = []
    y_train = []

    for i in range(days, training_data.shape[0]):
        X_train.append(training_data[i-days:i])
        y_train.append(training_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

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

    return X_train, y_train, X_test, y_test, scale


"""
Method that trains LSTM on stock data, returns the RMSE of the regression, and plots predicted vs actual test values
Arguments:
- X_train/y_train: training data and labels
- X_test/y_test: testing data and labels
- scale: scale used by scaler when preprocessing
- days: number of days used to predict stock prices
- hidden_layers: number of hidden layers in the LSTM
- nodes: list of integers detailing the number of nodes in each hidden_layer
- dropout: how much dropout occurs between layers (dropout is a random selection of nodes have their output become 0)
- training_epochs: how many times the training data will be run through for training the LSTM
"""

def predict_stocks_LSTM(X_train, y_train, X_test, y_test, scale, days, hidden_layers, nodes, dropout, training_epochs, show_model=True):

    model = Sequential()

    model.add(LSTM(units = days, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], X_train[0].shape[1])))
    model.add(Dropout(dropout))

    for i in range(hidden_layers):
        if (i == hidden_layers - 1):
            model.add(LSTM(units = nodes[i], activation = 'relu'))
        else:
            model.add(LSTM(units = nodes[i], activation = 'relu', return_sequences = True))
        model.add(Dropout(dropout))

    model.add(Dense(units = 1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=training_epochs, batch_size=32)

    y_pred = model.predict(X_test)
    
    y_pred = y_pred*scale
    y_test = y_test*scale

    if show_model:
        plt.figure(figsize=(14,5))
        plt.plot(y_test, color='red', label='Real Stock Price')
        plt.plot(y_pred, color='green', label='Predicted Stock Price')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    rmse = np.sqrt((1/(y_pred.size)) * np.sum(np.square(y_pred - y_test)))
    print(rmse)
    return rmse


def linear_regression(data, stock, feature):

    stock_data = data[data['ticker_symbol'] == stock].copy()

    #Drops features not in feature list
    dropped_features = [] 
    for i in range(4):
        if i not in feature:
            dropped_features.append(i+3)
    stock_data.drop(columns = stock_data.columns[dropped_features], axis=1, inplace=True)

    #Split data into testing and training, using 8 years and 2 years from Train/Test Split
    training_data_0 = stock_data[stock_data['day_date'] < '2019-01-01'].copy()
    testing_data = stock_data[stock_data['day_date'] >= '2019-01-01'].copy()
    
    training_data = training_data_0.drop(['ticker_symbol', 'day_date'], axis=1)
    testing_data = testing_data.drop(['ticker_symbol', 'day_date'], axis=1)
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    scale = 1/scaler.scale_[0]

    print(training_data)
    X_train = training_data[:,1]
    print(X_train)
    y_train = training_data[:,0]
    print(y_train)

    testing_data = scaler.transform(testing_data)
    X_test = testing_data[:,1]
    y_test = testing_data[:,0]

    linear_model = Sequential()
    linear_model.add(Dense(units=1))

    linear_model.compile(optimizer='adam', loss='mean_absolute_error')
    linear_model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.2)

    y_pred = linear_model.evaluate(X_test, y_test)

    x = np.linspace(0.0, 1, 100)
    y = linear_model.predict(x)

    plt.scatter(X_train, y_train)
    plt.plot(x, y, color='k')



#In Progress
def sentimentRegression(data, stock, tweet_ids, tweets):
    stock_data = data[data['ticker_symbol'] == stock].copy()
    stock_ids = tweet_ids[tweet_ids['ticker_symbol'] == stock]['tweet_id'].tolist()
    mask = tweets['tweet_id'].isin(stock_ids)
    stock_tweets = tweets.loc[mask]
    stock_tweets_strings = stock_tweets['body'].tolist()
    stock_tweets_dates = stock_tweets['post_date'].tolist()
    stock_tweets_dates = list(pd.to_datetime(stock_tweets_dates, unit='s').strftime('%Y-%m-%d'))
    twitter_data = list(zip(stock_tweets_strings, stock_tweets_dates))

    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    prev_date = twitter_data[0][1]
    avg_sentiment = 0
    total = 0
    for (tweet, date) in twitter_data:
        if (date != prev_date):
            avg_sentiment = avg_sentiment / total
            sentiments.append((avg_sentiment, prev_date))
            avg_sentiment = analyzer.polarity_scores(tweet)
            total = 1
        else:
            avg_sentiment += analyzer.polarity_scores(tweet)
            total += 1
        prev_date = date






