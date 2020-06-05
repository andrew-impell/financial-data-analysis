import glob
import numpy as np
import pandas as pd
import os
import pickle
import flair
from collections import Counter
from tqdm import tqdm

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')


def process_news(fname):
    print(f'\nReading {fname}...')

    with open(fname, 'rb') as f:
        papers = pickle.load(f)

    # Clean the headline list

    print('Initial Length: ', str(len(papers)))

    # papers = [i for i in papers if not ('Analyst Actions' in i)]

    papers = list(set(papers))

    print('Length without duplicates: ', str(len(papers)))

    papers[:] = [x for x in papers if x]

    print('Length cleaned: ', str(len(papers)))

    # Perform NLP

    sentiment = []
    polarity = []

    for article in tqdm(papers):
        s = flair.data.Sentence(article)
        flair_sentiment.predict(s)
        total_sentiment = s.labels
        [sent, pol] = str(total_sentiment[0]).split()
        pol = float(pol[pol.find('(')+1:pol.find(')')])
        sentiment.append(sent)
        polarity.append(pol)

    # Count number of each NEG or POS
    c = Counter(sentiment)
    total = sum(c.values())
    percent = {key: value/total * 100 for key, value in c.items()}

    print(percent)
    print(f'Average Polarity: {np.mean(polarity)}')

    return np.mean(polarity), percent


cwd = os.getcwd()

news_glob = glob.glob(cwd + '/news/news_*.pkl')

NLP_dict = {}

for news_file in tqdm(news_glob):
    end = news_file.find('.')
    start = end - 10
    date = news_file[start:end]
    polarity, percent = process_news(news_file)

    NLP_dict[date] = (polarity, percent)


df = pd.DataFrame.from_dict(data=NLP_dict, orient='index',
                            columns=['Polarity', 'Sentiment'])

df.index = pd.to_datetime(df.index)

print(df)

df.to_csv(cwd + '/news/sentiment_df.csv')
