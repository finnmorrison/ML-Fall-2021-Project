# Stock Price Prediction using Sentiment Analysis and Recurrent Neural Networks
## Introduction/Background
We hope to answer whether public opinion, as observed through Twitter tweets, can be used as an accurate predictor of stock market performance. Through sentiment analysis, we can take tweets about certain companies and judge their approximate sentiment, whether positive or negative. By comparing the sentiment of tweets about a company against that companies’ stock market performance at certain points in time, we might be able to establish some correlation.
## Problem Definition
Our group seeks to explore the relationship between the sentiment of tweets about a certain company and its competitors and the performance of said company's stock. The goal will be to see if the average daily sentiment of a company's stock is able to be used in addition to regular financial indicators to predict the closing price of a company's stock.
## Methods
The main method we have explored thus far is using a Recurrent Neural Network to build a model of stock price prediction. A Recurrent Neural Network (RNN) differs from a standard fully-connected Neural Network in that the previous output of a particular node will act as an input. This makes a RNN useful for time series data, where an input is dependent on previous inputs. In particular we are looking at a Long Short-Term Memory, LSTM, which is a specific architecture for an RNN where data at one time step can affect the neural network arbitrarily far in time. This is beneficial for stock data in which longer time periods than just the previous day stock price.

The stock dataset [5] that we are taking advantage of includes the close value, open value, volume, high value, and low value of the stock for Apple, Microsfot, Tesla, Google, and Amazon from 2010-2020. This dataset forms the basis of our financial data that we draw from. In addition we are using data from a collection of tweets [6] about these companies from 2015-2020. Using Vader, we are computing the average daily sentiment of these tweets for each company. This forms our sentiment feature that we use along our financial features.

Since the dataset is time-dependnet, we use data before January 1, 2019 as training, and all data in 2019 and 2020 as testing. When just looking at financial data this creates approxiamtely an 80/20 split in the data. When taking sentiment into consideration (which doesn't begin until 2015) this is closer to a 60/40 split for now, but this might be adjusted in the future.

## Results/Discussion

*For now I'm just going to do a rough outline of the results*

For selecting which features to use, we chose a simple architecture for our LSTM and ran cross-validation on all possible combinations of the stock features: Close Value, Volume, Open Value, High Value, Low Value, Avg Daily Sentiment. Due to the way we are currently structuring our code, Close Value is always used a feature.

The feature combinations were chosen based off which ones had the lowest average RMSE after performing 10-folds cross-validation on the training data. This was performed using the RNN archiecture of taking into consideration 30 previous days of data, 1 hidden layer with 30 nodes, and a dropout of 0.2.

Best Feature Combinations (Apple):
  1. (Close Value), Avg. Sentiment
  2. (Close Value), Open Value, High Value, Low Value, Avg. Sentiment
  3. (Close Value), Volume, Open Value, High Value
  4. (Close Value), High Value, Low Value, Avg. Sentiment
  5. (Close Value), Open Value, Avg. Sentiment

We can analyze how each of these combination of features doeson the test set:

1. (Close Value), Avg. Sentiment **RMSE: 906.9979436499955**
![1](https://user-images.githubusercontent.com/45157298/142055215-abe3763a-cde8-4f43-a8b8-871c1e8a88ec.png)

2. (Close Value), Open Value, High Value, Low Value, Avg. Sentiment **RMSE: 872.1925765828029**
![1234](https://user-images.githubusercontent.com/45157298/142055350-203a0fe2-d3d7-4dfb-86a8-a1ea80e41175.png)


3. (Close Value), Volume, Open Value, High Value **RMSE: 921.5304447191337**
![012](https://user-images.githubusercontent.com/45157298/142055363-208a7f1f-bd84-4ebd-a9e6-b7b5b148cc18.png)


4. (Close Value), High Value, Low Value, Avg. Sentiment **RMSE: 891.1052655709665**
![234](https://user-images.githubusercontent.com/45157298/142055388-c398eb6a-b090-4e3e-995c-d1d61e85b0f4.png)


5. (Close Value), Open Value, Avg. Sentiment **856.0626119226954**
![14](https://user-images.githubusercontent.com/45157298/142055411-329a4de5-3f79-4463-b47d-faf773b195a9.png)



GOOGLE

The combinations below were chosen based off which ones had the lowest average RMSE after performing 10-folds cross-validation on the training data where the number of previous days of data taken into consideration varied from 5 to 40 in increments of 5. This was performed using the RNN archiecture of taking into consideration 1 hidden layer with the number of nodes dependent on the number of days, as well as a dropout of 0.2.

Best Day / Feature Combinations (Google):
1.	Days = 10 / Features = (Close Value), Volume
2.	Days = 10 / Features = (Close Value), Volume, High Value, Low Value
3.	Days = 10 / Features = (Close Value), Volume, Open Value, High Value, Low Value
4.	Days = 15 / Features = (Close Value), Volume, Open Value
5.	Days = 20 / Features = (Close Value), Volume, Low Value


We can analyze how each of these combination of features does on the test set:

1.	Days = 10 / Features = (Close Value), Volume **RMSE: 2294.413724748646**
![download](https://user-images.githubusercontent.com/72471609/142086499-121f9622-b8e9-4c6c-92fa-b598d3df9912.png)


2.	Days = 10 / Features = (Close Value), Volume, High Value, Low Value **RMSE: 2299.369869822851**
![download-5](https://user-images.githubusercontent.com/72471609/142086523-01caa8bb-187c-4843-9081-b4cf28c1c52c.png)


3.	Days = 10 / Features = (Close Value), Volume, Open Value, High Value, Low Value **RMSE: 2294.8827952285183**
![download-6](https://user-images.githubusercontent.com/72471609/142086539-0a2e3c0b-c656-4f9d-a206-45250b6c821e.png)


4.	Days = 15 / Features = (Close Value), Volume, Open Value **RMSE: 2304.103944690263**
![download-3](https://user-images.githubusercontent.com/72471609/142086581-716ef3b3-a7a0-480d-995d-a1ad061d3ef9.png)


5.	Days = 20 / Features = (Close Value), Volume, Low Value **RMSE: 2316.265447861915**
![download-7](https://user-images.githubusercontent.com/72471609/142086592-c3b120e7-2d91-44b3-a217-47530cdb1b7a.png)


Worst Day / Feature Combinations (Google):
1.	Days = 5 / Features = (Close Value), Open Value, Low Value, **RMSE: 3680.8685713126324**
2.	Days = 5 / Features = (Close Value), Volume, High Value, **RMSE: 3054.2719010622286**
3.	Days = 5 / Features = (Close Value), Low Value, **RMSE: 2947.300821742468**
4.	Days = 5 / Features = (Close Value), **RMSE: 2916.9163665894507**
5.	Days = 5 / Features = (Close Value), High Value, **RMSE: 2746.198119766924**


## References

[1] Using Twitter Attribute Information to Predict the Stock Market, https://arxiv.org/abs/2105.01402

[2] k-Shape: Efficient and Accurate Clustering of Time Series, http://people.cs.uchicago.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf

[3] Effectiveness of Artificial Intelligence in Stock Market Prediction based on Machine Learning, https://arxiv.org/abs/2107.01031

[4] VADER, https://github.com/cjhutto/vaderSentiment

[5] Values of Top NASDAQ Companies from 2010 to 2020, https://www.kaggle.com/omermetinn/values-of-top-nasdaq-copanies-from-2010-to-2020

[6] Tweets about the Top Companies from 2015 to 2020, https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020

[7] Understanding LSTM Networks, https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Project Member Responsibilities
Christopher Raulston - Mining/Analyzing Tweets \
Finn Morrison - Hierarchical Clustering \
William Sheppard - Sentimental Analysis \
Caleb Partin - Predictive Model \
Mohit Aggarwal - Analysis of Results


The feature combinations were chosen based off which ones had the lowest average RMSE after performing 10-folds cross-validation on the training data. This was performed using the RNN archiecture of taking into consideration 30 previous days of data, 1 hidden layer with 30 nodes, and a dropout of 0.2.

Best Feature Combinations (Amazon):
  1. (Close Value), High Value, Low Value, **RMSE: 37834.282204068244**
  2. (Close Value), Volume, Open Value, High Value, **RMSE: 37921.89313297843**
  3. (Close Value), High Value, **RMSE: 37934.6007375283**
  4. (Close Value), Open Value, **RMSE: 38139.58991192261**
  5. (Close Value), Volume, Open Value, Low Value, **RMSE: 38282.51159986677**


We can analyze how each of these combination of features does on the test set:

  1. (Close Value), High Value, Low Value, **RMSE: 37834.282204068244**
![image](https://user-images.githubusercontent.com/47957718/142135156-e8131463-c8f3-41cf-9b16-e4c6c498c6b6.png)

  3. (Close Value), Volume, Open Value, High Value, **RMSE: 37921.89313297843**
![image](https://user-images.githubusercontent.com/47957718/142135163-1a1e56a8-ccb7-4ec3-b58a-a034e50127cb.png)

  5. (Close Value), High Value, **RMSE: 37934.6007375283**
![image](https://user-images.githubusercontent.com/47957718/142135172-51696ff9-89aa-483f-b5c7-5c92fb826458.png)

  7. (Close Value), Open Value, **RMSE: 38139.58991192261**
![image](https://user-images.githubusercontent.com/47957718/142135176-5e576123-8278-4582-b3f1-d137cc51d44d.png)

  8. (Close Value), Volume, Open Value, Low Value, **RMSE: 38282.51159986677**
![image](https://user-images.githubusercontent.com/47957718/142135193-c2905446-8e1d-4638-9095-3239815984aa.png)
