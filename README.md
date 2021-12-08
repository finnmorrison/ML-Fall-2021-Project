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

### LSTM

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



For Google, the combinations below were chosen based off which ones had the lowest average RMSE after performing 10-folds cross-validation on the training data where the number of previous days of data taken into consideration varied from 5 to 40 in increments of 5. This was performed using the RNN archiecture of taking into consideration 1 hidden layer with the number of nodes dependent on the number of days, as well as a dropout of 0.2.

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



Best Feature Combinations (Tesla):

1)  (Close Value), Volume, High Value, low Value, **RMSE: 4776.0253458907755**
2)  (Close Value), Volume, low Value, **RMSE: 4866.940645985312**
3)  (Close Value), High Value, low Value, **RMSE: 4935.57234190576**
4)  (Close Value), low Value, **RMSE: 4974.552449942653**
5)  (Close Value), Open Value, High Value, **RMSE: 5026.822804101358**




1)  (Close Value), Volume, High Value, low Value, **RMSE: 4776.0253458907755**

<img width="847" alt="image" src="https://user-images.githubusercontent.com/61914353/142138658-c4684d65-7405-4db6-b67b-841ae6c638a1.png">


2)  (Close Value), Volume, low Value, **RMSE: 4866.940645985312**

<img width="847" alt="image" src="https://user-images.githubusercontent.com/61914353/142138846-37418f22-2dbd-427c-b794-3282b1c84c59.png">

3)  (Close Value), High Value, low Value, **RMSE: 4935.57234190576**

<img width="847" alt="image" src="https://user-images.githubusercontent.com/61914353/142139115-dd4147ed-642d-4be0-825c-53b12c36e8fe.png">

4)  (Close Value), low Value, **RMSE: 4974.552449942653**

<img width="847" alt="image" src="https://user-images.githubusercontent.com/61914353/142139234-5251c6a7-2890-4aff-bb0b-9583d650ddbd.png">

5)  (Close Value), Open Value, High Value, **RMSE: 5026.822804101358**

<img width="847" alt="image" src="https://user-images.githubusercontent.com/61914353/142139320-9b2d50c0-80b3-4ec7-af46-c62a82d1d73d.png">


### Sentiment Linear Regression

A simple yet naive way to examine how sentiment values affect a company's stock price is to see if there is a linear correlation between average daily sentiment and the closing price of a company's stock. To this extenet we performed a linear regression between closing price and average daily sentiment, using the polarity score given by vaderSentiment:

Apple:

![image](https://user-images.githubusercontent.com/45157298/145050258-59fb2533-3bed-4301-811b-df26ab29f621.png)

*R^2 value: 0.2813197875473433*

Tesla:

![image](https://user-images.githubusercontent.com/45157298/145050286-b68e2aad-da78-4ff5-9dad-531969ce6caa.png)

*R^2 value: 0.04001449241679578*

Microsoft:

![image](https://user-images.githubusercontent.com/45157298/145050328-19de6a68-f787-45ff-bdfd-c52ae3a78315.png)

*R^2 value: 0.20991147468090765*

Amazon:

![image](https://user-images.githubusercontent.com/45157298/145050375-899f1d29-8d76-42ab-b8f4-d8850d13cb37.png)

*R^2 value: 0.24346021363669412*

Google:

![image](https://user-images.githubusercontent.com/45157298/145050405-808adba8-179b-4b6a-b1b8-7f10f72c6b8f.png)

*R^2 value: 0.0011274365555208332*

We can extend this analysis by considering a lag component to the linear regression. That is, we associate the average daily sentiment of one day with the stock price a certain number of days later. Let's look at two companies that already have some positive correlation in regression, in this case Apple and Amazon. And we will look at how the R^2 value changes when we change this lag variable.

Apple:
- 0 Day Lag: 0.2813197875473433
- 5 Day Lag: 0.27530696615020944
- 10 Day Lag: 0.2722549818900133
- 15 Day Lag: 0.2734955583157106
- 20 Day Lag: 0.27152019709246167
- 25 Day Lag: 0.2717001177102448
- 30 Day Lag: 0.2720236359406325
- 35 Day Lag: 0.27729167630505647
- 40 Day Lag: 0.2862514252654186
- 45 Day Lag: 0.29268776795477913
- 50 Day Lag: 0.29711182883935605
- 55 Day Lag: 0.30178280360582266
- **60 Day Lag: 0.311770734397239**

Amazon:
- **0 Day Lag: 0.24346021363669412**
- 5 Day Lag: 0.23609447615100743
- 10 Day Lag: 0.23044712975079973
- 15 Day Lag: 0.23164063741307594
- 20 Day Lag: 0.2270248051323045
- 25 Day Lag: 0.21987060376916856
- 30 Day Lag: 0.2174625788266198
- 35 Day Lag: 0.21892752288376172
- 40 Day Lag: 0.21784266747176873
- 45 Day Lag: 0.21741909672585935
- 50 Day Lag: 0.22217990628041973
- 55 Day Lag: 0.2166414388138581
- 60 Day Lag: 0.2174400895944305

The two things of note from this analysis is how for Apple, the highest correlation is achieved with a two month lag. This shows possibly that certain sentiments and feelings about apple stock did not manifest in the stock price until two months later. The other intesting thing to note is that changing the lag does not heavily influence the regression for either company. At least not in this two month window. This could hint at the fact that for many companies, how general sentiment influences the price can be a gradual process and may take many months to manifest in terms of the price.

### Feature Correlation

A simple way to view how our features relate to one one another is to calculate Pearson correlation coefficients across the different features. We are primarily concerned with how sentiment is related to closing volume, though it is also interesting to see how it relates to volume, and volume, to other features respectively. We noticed from sentiment regression analysis that for TSLA and GOOGL, closing price was weakly related to sentiment; this pattern was confirmed here.

AAPL:

<img width="527" alt="image" src="https://user-images.githubusercontent.com/61914353/145162040-c8931008-b8e4-4db5-800a-25d8d4ee36fd.png">


TSLA:

<img width="527" alt="image" src="https://user-images.githubusercontent.com/61914353/145162110-42ecc142-a8c7-474f-9e51-d12e257d1c4e.png">

MSFT:

<img width="527" alt="image" src="https://user-images.githubusercontent.com/61914353/145162154-3ea8c5c9-1d8b-41a5-b84d-56f2897af235.png">


AMZN:

<img width="527" alt="image" src="https://user-images.githubusercontent.com/61914353/145162188-b39475df-fb6e-4d2e-9349-4347f4f787b3.png">


GOOGL:

<img width="527" alt="image" src="https://user-images.githubusercontent.com/61914353/145162208-0ab060e3-dd5d-48bf-b442-65c5e0b500f0.png">




## Conclusions

The impact of sentiment on a company's stock price are complex and varied, and often times it seems to also depend on the company. When determining which financial features are best to use in a predictive model, the results are highly varied. Between the 6 financial indicators given to us in the data, the various combinations of them to predict closing price could differ in their usefulness between the major companies we explored. However, when including the sentiment values of tweets in Apple Stock, we instantly see that this feature (alongside closing price) becomes one of the most useful to consider in our model.

It's interesting to compare this with the results of the linear regression. In general the companies that did have some correlation between sentiment and stock price (Apple, Amazon, Microsoft), still had a small enough signifiance to delcare that there wasn't much correlation. This displays that the relationship between stock price and sentiment is significant yet complex and highly non-linear. It is a relationship that can only be captured through the non-linearity of a neural network, and isn't plain to see through simple regression. It's possible that changes in sentiment can have immediate or long-term impact, and the multitude of factors determining this impact are not easily captured through a simple predictive algorithm.


## References

[1] Using Twitter Attribute Information to Predict the Stock Market, https://arxiv.org/abs/2105.01402

[2] k-Shape: Efficient and Accurate Clustering of Time Series, http://people.cs.uchicago.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf

[3] Effectiveness of Artificial Intelligence in Stock Market Prediction based on Machine Learning, https://arxiv.org/abs/2107.01031

[4] VADER, https://github.com/cjhutto/vaderSentiment

[5] Values of Top NASDAQ Companies from 2010 to 2020, https://www.kaggle.com/omermetinn/values-of-top-nasdaq-copanies-from-2010-to-2020

[6] Tweets about the Top Companies from 2015 to 2020, https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020

[7] Understanding LSTM Networks, https://colah.github.io/posts/2015-08-Understanding-LSTMs/




