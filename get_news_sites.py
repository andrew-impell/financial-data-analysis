import pickle
import newspaper
import time
'''
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

cat_list = []

for site in sites:
    try:
        first, last = site.find('.'), site.rfind('.')
        print(f'\nGetting {site[first+1:last].capitalize()}...')
        paper = newspaper.build(site, language='en')
        for category in paper.category_urls():
            print(category)
            cat_list.append(category)
        time.sleep(0.1)
    except Exception as e:
        print(e)


keywords = ['technology', 'stocks',
            'business', 'economy',
            'commodities', 'money',
            'politics', 'markets']


good_links = []

for keyword in keywords:
    for cat in cat_list:
        if keyword in cat:
            good_links.append(cat)

with open('good_links.pkl', 'wb') as f:
    pickle.dump(good_links, f)
'''
with open('good_links.pkl', 'rb') as f:
    goods = pickle.load(f)
print(goods)
