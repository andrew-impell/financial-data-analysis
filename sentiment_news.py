import newspaper
import time
import pickle
from textblob import TextBlob
import numpy as np
import flair
from collections import Counter
import os
from datetime import date

today = date.today()
cwd = os.getcwd()

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

# Market Beat
sites = [
    'https://www.nytimes.com/section/business',
    'https://www.marketwatch.com/',
    'https://www.wsj.com/news/business',
    'https://www.wsj.com/news/markets',
    'https://www.wsj.com/news/politics',
    'https://fivethirtyeight.com/',
    'https://www.reuters.com/',
    'https://www.ft.com/',
    'https://www.washingtonpost.com/business/',
    'https://www.bloomberg.com/',
    'https://www.cbc.ca/news/business',
    'https://www.bbc.com/news/business',
    'https://news.yahoo.com/business/',
    'https://www.nbcnews.com/business',
    'https://www.cnn.com/BUSINESS',
    'http://forbes.com',
    'https://www.thestreet.com/',
    'https://www.cnbc.com/stocks/'
]


with open('good_links.pkl', 'rb') as f:
    sub_links = pickle.load(f)

sites.append(sub_links)


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


most_len = len(' Channels &amp; Frequencies')

papers = []
site_dict = {}

for site in sites:
    try:
        first, last = site.find('.'), site.rfind('.')
        print(f'\nGetting {site[first+1:last].capitalize()}...')
        paper = newspaper.build(site, language='en')
        articles = paper.articles
        print(f'[+] Articles: {len(articles)}\n')
        for article in articles:
            article.download()
            article.parse()
            raw_text = str(article.title)
            cleaned = raw_text.rstrip().strip()
            print('\t', cleaned)
            if isEnglish(cleaned) and cleaned != 'None':
                if len(cleaned) > most_len:
                    papers.append(cleaned)
        time.sleep(0.1)
    except Exception as e:
        print(e)


fname = cwd + '/news/news_' + str(today) + '.pkl'

with open(fname, 'wb') as f:
    pickle.dump(papers, f)


with open(fname, 'rb') as f:
    papers = pickle.load(f)

print('Initial Length: ', str(len(papers)))

# papers = [i for i in papers if not ('Analyst Actions' in i)]

papers = list(set(papers))

print('Length without duplicates: ', str(len(papers)))

papers[:] = [x for x in papers if x]

print('Length cleaned: ', str(len(papers)))


sentiment = []
polarity = []

for article in papers:
    s = flair.data.Sentence(article)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    [sent, pol] = str(total_sentiment[0]).split()
    pol = float(pol[pol.find('(')+1:pol.find(')')])
    sentiment.append(sent)
    polarity.append(pol)

c = Counter(sentiment)
total = sum(c.values())
percent = {key: value/total * 100 for key, value in c.items()}

print(percent)
print(f'Average Polarity: {np.mean(polarity)}')
