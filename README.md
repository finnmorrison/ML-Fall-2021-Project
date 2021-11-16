# Stock Price Prediction using Sentiment Analysis and Recurrent Neural Networks
## Introduction/Background
We hope to answer whether public opinion, as observed through Twitter tweets, can be used as an accurate predictor of stock market performance. Through sentiment analysis, we can take tweets about certain companies and judge their approximate sentiment, whether positive or negative. By comparing the sentiment of tweets about a company against that companies’ stock market performance at certain points in time, we might be able to establish some correlation.
## Problem Definition
Our group seeks to explore the relationship between the sentiment of tweets about a certain company and its competitors and the performance of said company's stock. We will first try to see how we can use time series clustering algorithms to correlate different stock prices and create groups of correlated companies. Then using a sentiment analyzer to determine the tweets about this group of companies, we will use regression techniques to correlate the sentiments with stock performance.
## Methods
Our pipeline is split into stages: clustering, sentiment analysis and a predictive model. Start by using hierarchal clustering to divide companies into different clusters. Then, we will use the Twitter API and Python libraries for sentiment analysis-- looking at the tweets of similar companies through verified accounts or accounts with more than a set amount of followers. Then, we will train our predictive model, which will either be a regression model or a neural net.

The main method we have explored thus far is using a Recurrent Neural Network to build a model of stock price prediction. A Recurrent Neural Network (RNN) differs from a standard fully-connected Neural Network in that the previous output of a particular node will act as an input. This makes a RNN useful for time series data, where an input is dependent on previous inputs. In particular we are looking at a Long Short-Term Memory, LSTM, which is a specific architecture for an RNN where data at one time step can affect the neural network arbitrarily far in time. This is beneficail for stock data in which longer time periods than just the previous day stock price 


## Potential Results/Discussion
By looking at company stock data on twitter, we would hope to accurately predict the stock prices of correlated companies. While we do expect there to be a margin of error in the predicted prices, we plan to minimize this by correlating companies from our training dataset in a precise manner. Overall, researching tweets to predict stock market changes could very well encourage new involvement within the stock market. A successful prediction could have significant implications; by determining the movement of future stock prices, we would be able to increase profit opportunities for potential investors.

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
