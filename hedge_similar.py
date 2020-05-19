import matplotlib.pyplot as plt
from matplotlib import style
import pandas_datareader as pdr
from collections import Counter
import pickle
import numpy as np
import pandas as pd
import urllib.request
import os
import re

style.use('ggplot')

cwd = os.getcwd()
'''
user_agent = 'Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; Lumia 640 XL LTE) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Mobile Safari/537.36 Edge/14.14393'
headers = {'User-Agent': user_agent, }


base_url = 'https://fintel.io'

data = '/home/andrew/Documents/dev/data/hedge_fund_list.txt'

with open(data) as s:
    html_list = s.readlines()

link_pattern = r'"([A-Za-z0-9_\./\\-]*)"'

link_list = []
for h in html_list:
    m = re.search(link_pattern, h)
    link_list.append(m.group().strip('""'))


funds = []

for fund in link_list:
    try:
        request = urllib.request.Request(base_url + fund, None, headers)
        response = urllib.request.urlopen(request)
        data = response.read()  # The data u need
        dfs = pd.read_html(data)
        funds.append(dfs[1])
        sleep(5)
    except:
        print(fund)

with open('funds_13F.pkl', 'wb') as f:
    pickle.dump(funds, f)
'''


def get_same(a, b):
    # a, b are lists
    if set(a) != set(b):
        same = set(a) & set(b)
        return list(same)
    else:
        return []


with open('funds_13F.pkl', 'rb') as f:
    funds = pickle.load(f)

tickers = []

for f in funds:
    cleaned = []
    securities = f.iloc[:, 2].values
    for s in securities:
        idx = s.find('/')
        cleaned.append(s[:idx-1])
    tickers.append(cleaned)

set_tickers = [set(sublist) for sublist in tickers]
flat_tick = [item for sublist in set_tickers for item in sublist]
c = Counter(flat_tick)
most = [(i, c[i] / len(flat_tick) * 100.0) for i, count in c.most_common()]
'''
df_list = []

for ticker, p in most[:50]:
    try:
        s = pdr.get_data_yahoo(ticker, start='2015-01-01')
        close = s['Close']
        df_list.append(close)
    except:
        print(ticker)

hedge_df = pd.concat(df_list, axis=1)
hedge_df.to_csv('hedge_df.csv')
'''
spy = pdr.get_data_yahoo('SPY', start='2015-01-01')

hedge_df = pd.read_csv('hedge_df.csv')


weightx = np.arange(1, hedge_df.values.shape[1])
weighty = [-0.01*x+0.5 for x in weightx]

w_hedge = []
for i in range(hedge_df.values.shape[0]):
    hedges = hedge_df.values[i, 1:]
    weighty = np.array(weighty).T
    s = np.sum(hedges * weighty)
    w_hedge.append(s)

hedge_index_w = pd.DataFrame(w_hedge)
hedge_returns_w = hedge_index_w.pct_change().cumsum()

hedge_index = hedge_df.sum(axis=1)
hedge_returns = hedge_index.pct_change().cumsum()

spy_returns = spy['Close'].pct_change().cumsum()
print(spy_returns)
plt.plot(spy_returns.index, spy_returns, label='SPY')
plt.plot(spy_returns.index, hedge_returns_w.values, label='Hedge Fund Picks (Weighted)')
plt.plot(spy_returns.index, hedge_returns.values, label='Hedge Fund Picks (Unweighted)')
plt.legend()
plt.xlabel('Date')
plt.ylabel('% Return')
plt.title('Hedge Fund Picks vs S&P 500')
plt.show()

'''
same_stocks = []

for i in tickers:
    for j in tickers:
        same_stocks.append(get_same(i, j))

same_of_same = []

for i in same_stocks:
    for j in same_stocks:
        same_of_same.append(get_same(i, j))

clean_same = [x for x in same_of_same if x != []]
flat_list = [item for sublist in clean_same for item in sublist]

print(set(flat_list))
'''
