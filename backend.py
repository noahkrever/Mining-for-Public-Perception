import tweepy
from bs4 import BeautifulSoup
import requests
import json
from textblob import TextBlob
import sys
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
# import pycountry
import re
import string
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pygooglenews import GoogleNews
from newspaper import Article
import gender_guesser.detector as gender
from tweepy_keys import *

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

def twitter_sentiment_analysis(word, n): #returns list of polarities with (neg, neut, pos) counts

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    polarities = []

    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    tweets = tweepy.Cursor(api.search, q = word).items(n)

    positive = 0
    neutral = 0
    negative = 0
    polarity = 0
    subjectivity = 0


    for tweet in tweets:
        tweet_list.append(tweet.text)
        analysis = TextBlob(tweet.text)
        polarity += analysis.sentiment.polarity
        # subjectivity += analysis.sentiment.subjectivity
        # print(analysis.sentiment.polarity)

        if analysis.sentiment.polarity == 0:
            # neutral += 1
            neutral_list.append(tweet.text)
        elif analysis.sentiment.polarity < 0:
            # negative += 1
            negative_list.append(tweet.text)
        elif analysis.sentiment.polarity > 0:
            # positive += 1
            positive_list.append(tweet.text)

        # print("Out of the " + str(n) + " tweets containing '" + str(searchTerm) + "',")
        # print(str(neutral) + " were neutral")
        # print(str(negative) + " were negative")
        # print(str(positive) + " were positive")
        # print("Polarity is " + str(polarity))

        # polarities.append((polarity/n,negative,neutral,positive)))
    tweet_list = pd.DataFrame(tweet_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)

    tweet_list.drop_duplicates(inplace = True)

    tw_list = pd.DataFrame(tweet_list)
    tw_list['text'] = tw_list[0]
    #Removing RT, Punctuation etc
    # remove_rt = lambda x: re.sub('RT @\w+: ',' ',x)
    remove_rt = lambda x: re.compile('RT @').sub('@', x, count=1)
    rt = lambda x: re.sub('(@[A-Za-z0–9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)',' ',x)
    # tw_list['text'] = tw_list.text.map(remove_rt).map(rt)
    tw_list['text'] = tw_list.text.str.lower()
    tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in tw_list['text'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        if neg > pos:
            tw_list.loc[index, 'sentiment'] = 'negative'
        elif pos > neg:
            tw_list.loc[index, 'sentiment'] = 'positive'
        else:
            tw_list.loc[index, 'sentiment'] = 'neutral'
            tw_list.loc[index, 'neg'] = neg
            tw_list.loc[index, 'neu'] = neu
            tw_list.loc[index, 'pos'] = pos

    return tw_list

# test = twitter_sentiment_analysis('Biden',100)

def news_sentiment_analysis(word,n): #query, num articles
    titles = []
    keywords = []
    polarities = []
    subjectivities = []
    links = []
    texts = []
    gn = GoogleNews(lang = 'en', country = 'US')
    s = gn.search(word)
    i = 0
    for article in s['entries']:
        if i >= n:
            break
        try:
            
            a = Article(url=article['link'])
            a.download()
            a.parse()
            titles.append(article['title'])
            links.append(article['link'])
            texts.append(a.text)
            a.nlp()
            keywords.append(a.keywords)
            analysis = TextBlob(a.text)
            polarities.append(analysis.sentiment.polarity)
            subjectivities.append(analysis.sentiment.subjectivity)
        except:
            continue
        i += 1
    di = {'title':titles, 'text':texts, 'keywords':keywords, 'polarity':polarities, 'subjectivity':subjectivities}
    return pd.DataFrame.from_dict(di)

# sa = news_sentiment_analysis('Goldman Sachs',20)
# print(sa['polarity'])
# print(sa['keywords'])

def getCompanyInfo(word,n_twitter,news_n):
    twit_df = twitter_sentiment_analysis(word,n_twitter)
    news_df = news_sentiment_analysis(word,news_n)

    twit_text = list(twit_df['text'])
    twit_polarity = list(twit_df['polarity'])
    twit_subjectivity = list(twit_df['subjectivity'])

    
    news_text = list(news_df['text'])
    news_title = list(news_df['title'])
    news_polarity = list(news_df['polarity'])
    news_subjectivity = list(news_df['subjectivity'])
    news_keywords = list(news_df['keywords'])


    average_twit_polarity = np.mean(twit_polarity)
    average_news_polarity = np.mean(news_polarity)

    average_polarity = (average_twit_polarity + average_news_polarity)/2

    average_twit_subjectivity = np.mean(twit_subjectivity)
    average_news_subjectivity = np.mean(news_subjectivity)

    try:
        tw_max_ind = np.argpartition(twit_polarity, -3)[-3:]
        news_max_ind = np.argpartition(news_polarity, -3)[-3:]

        tw_max_polarities = ' '.join([str(twit_polarity[i]) for i in tw_max_ind])
        news_max_polarities = ' '.join([str(news_polarity[i]) for i in news_max_ind])

        tw_max_text = ', '.join([str(twit_text[i]) for i in tw_max_ind])
        news_max_title = ', '.join([str(news_title[i]) for i in news_max_ind])

        news_max_keywords = [news_keywords[i] for i in news_max_ind]

        tw_min_ind = np.argpartition(twit_polarity, 3)[:3]
        news_min_ind = np.argpartition(news_polarity, 3)[:3]

        tw_min_polarities = ' '.join([str(twit_polarity[i]) for i in tw_min_ind])
        news_min_polarities = ' '.join([str(news_polarity[i]) for i in news_min_ind])

        tw_min_text = ', '.join([str(twit_text[i]) for i in tw_min_ind])
        news_min_title = ', '.join([str(news_title[i]) for i in news_min_ind])

        news_min_keywords = [news_keywords[i] for i in news_min_ind]
    except:

        tw_max_polarities = None
        news_max_polarities = None

        tw_max_text = None
        news_max_title = None

        news_max_keywords = None

        tw_min_polarities = None
        news_min_polarities = None

        tw_min_text = None
        news_min_title = None

        news_min_keywords = None

    return {'business':word,'avg_twit_polarity':average_twit_polarity, 'avg_news_polarity':average_news_polarity,'avg_polarity':average_polarity,'avg_twit_subjectivity':average_twit_subjectivity, 'avg_news_subjectivity':average_news_subjectivity, 'max_twit_polarities':tw_max_polarities, 'max_twit_text':tw_max_text, 'max_news_polarities':news_max_polarities, 'max_news_title':news_max_title, 'max_news_keywords':news_max_keywords, 'min_twit_polarities':tw_min_polarities, 'min_twit_text':tw_min_text, 'min_news_polarity':news_min_polarities, 'min_news_title':news_min_title, 'min_news_keywords':news_min_keywords}
# df = pd.read_csv(r'C:\Users\noahk\Stat 3106\fortune1000.csv')
# print(df)

def compileCompanyDataset():
    li = []
    df = pd.read_csv(r'C:\Users\noahk\Stat 3106\fortune1000_2021.csv')
    i = 0
    for name in list(df['company'])[:100]:
        li.append(getCompanyInfo(name,250,25))
        print(i)
        i+=1
    return pd.DataFrame(li)

# df = compileCompanyDataset()
# print(df)
# df.to_csv('company_sa_21.csv')

def getGenderColumn():
    # df = pd.read_csv(r'C:\Users\noahk\Stat 3106\fortune1000_2021.csv')
    # name =df['CEO']

    # d = gender.Detector()
    # ceo_gender = [d.get_gender(n.split(' ')[0]) for n in name]
    
    # for i in range(len(ceo_gender)):
    #     if ceo_gender[i] in ['male', 'mostly_male']:
    #         ceo_gender[i] = 1
    #     elif ceo_gender[i] in ['female', 'mostly_female']:
    #         ceo_gender[i] = 0
    #     elif ceo_gender[i] == 'andy':
    #         ceo_gender[i] = np.random.choice([1,0])
    #     else:
    #         ceo_gender[i] = np.nan
    # newdf = pd.DataFrame()
    # newdf['ceo_gender'] = ceo_gender

    df = pd.read_csv(r'C:\Users\noahk\Stat 3106\fortune1000_2021.csv')
    woman =df['ceo_woman']
    ceo_gender = [x for x in range(len(woman))]
    for i in range(len(woman)):
        if woman[i] == 'yes':
            ceo_gender[i] = 0
        else:
            ceo_gender[i] = 1
    newdf = pd.DataFrame()
    newdf['ceo_gender'] = ceo_gender
    return newdf
    
df = getGenderColumn()
print(df)
df.to_csv(r'C:\Users\noahk\Stat 3106\ceo_gender21.csv')