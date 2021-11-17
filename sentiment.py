# from textblob import textblob; # NLP library
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; # sentiment anaysis specific library
'''
This file is a very rough start to sentiment analysis of the 2015-2020 dataset of tweets.
We can also make use of the Twitter API through through a dev account to really spice things up.
'''

import tweepy
# from textblob import TextBlob
# from worldcloud import WordCloud
import pandas as pd
# import numpy as np
import re
# import matplotlib.pyplot as plt
import requests # make HTTP requests to the Twitter endpoints
# import search-tweets-python # there will be issues looking at accounts with greater than 100 tweets per week

MAIN_DF = pd.read_csv("Tweet_truncated.csv")
# TODO: determine appropriate regular expressions to best filter tweet bodies by company
# perhaps we can add to these, but the dataset said tweets should contain the ticker so this should suffice
AAPL_RE = re.compile(r'.*\$AAPL.*', re.IGNORECASE)
# Google is unique because it has a class A and class C ticker that we mentioned in our meeting
GOOGL_RE = re.compile(r'.*\$GOOG[L]?.*', re.IGNORECASE) 
MSFT_RE = re.compile(r'.*\$MSFT.*', re.IGNORECASE)
AMZN_RE = re.compile(r'.*\$AMZN.*', re.IGNORECASE)
TSLA_RE = re.compile(r'.*\$TSLA.*', re.IGNORECASE)

'''
Filter the given dataframe passed in to find only those tweets mentioning a certain company/companies based on the RE passed in

ex: findByRegEx(df, 'body', r')
'''
def findByRegEx(columnName, regularExpression, df = MAIN_DF):
    return df[columnName].str.findall(regularExpression)
    

# 3rd column is date
# 4th column is body

# ALSO of interest to us
# 5th column comment_num
# 6th column retweet_num
# 7th column like_num

# VARY features and days, use same number of nodes as days we are using, do feature selection with the features

# Some utilities if we want to use the Twitter API for something
# twitter Developer account credentials
'''
account name: FinnMorrison19
password: MLProjectTeam!
Token and secret for later
'''

'''
Use of the Twitter API after developer account approval requires
a token and a secret, which can be stored in a yaml file
'''

def create_twitter_url(handle, maxResults):
    max_results_format = "max_results={}".format(maxResults)
    # note, may want to adjust this query later, so perhaps pass a dictionary of args
    # excluding tweets with media and other fields/expansions in this method, see Twitter API docs
    query = "query=from:{}".format(handle)
    url = "https://api.twitter.com/2/tweets/search/recent?{}&{}".format(max_results_format, query)
    return url

def main():
    # test regex
    # regEx = re.compile(r'\$AAPL', re.IGNORECASE)
    x = findByRegEx('body', AAPL_RE)
    print(x)
    # should return any tweet in full mentioning AAPL regardless of case, o/w the row in the dataframe is empty

if __name__ == "__main__":
    main()
    