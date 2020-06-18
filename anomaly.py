from fbprophet import Prophet
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
from tqdm import tqdm
import os
import math
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from matplotlib import style

style.use('ggplot')
cwd = os.getcwd()
np.seterr(divide='ignore')

'''
# Uncomment to get data

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2017, 3, 6)

russell = pd.read_excel('Russell-2000-Stock-Tickers-List.xlsx')

symbols = list(russell.iloc[:, 0].values[3:])

closes = []
volume = []
except_symbols = []
for symbol in tqdm(symbols):
    try:
        vdata = pdr.get_data_yahoo(symbol, start, end)
        cdata = vdata[['Close']]
        closes.append(cdata)
        vdata = vdata[['Volume']]
        volume.append(vdata)
    except Exception as e:
        print(e)
        except_symbols.append(symbol)
    time.sleep(0.2)


final_symbols = list(set(symbols) - set(except_symbols))

volume_df = pd.concat(volume, axis=1)
volume_df.columns = final_symbols


close_df = pd.concat(closes, axis=1)
volume_df.columns = final_symbols


volume_df.to_csv(cwd + '/volume_2000.csv')
close_df.to_csv(cwd + '/closes_2000.csv')


import glob


def get_stock_df():
    df_list = []
    df_dict = {}
    base = '/home/andrew/Documents/dev/econ/Stocks/'
    fname_glob = glob.glob(base + '*.txt')
    for file in tqdm(fname_glob):
        end = file.find('.us')
        start = file.rfind('/')
        ticker = file[start+1:end]
        try:
            ticker_df = pd.read_csv(file)
            ticker_df.index = pd.to_datetime(ticker_df['Date'])
            # ticker_df = ticker_df.rename({'Close': ticker})
            ticker = ticker.strip("\'")
            df_dict[ticker] = ticker_df['Close']
            # df_list.append(ticker_df[ticker])
        except Exception as e:
            print(e)closes.columns = symbols

    # stock_df = pd.concat(df_list, axis=1)
    stock_df = pd.DataFrame.from_dict(df_dict)
    print(stock_df)
    stock_df.to_csv('all_stocks.csv')

# read in data and graph it

stock_df = pd.read_csv('all_stocks.csv')
stock_df.index = pd.to_datetime(stock_df['Date'])
stock_df = stock_df.iloc[-2000:, 1:]
stock_df.fillna(0, inplace=True)

stock_df['average'] = np.exp(stock_df.sum(axis=1).cumsum().values)
print(stock_df['average'])

plt.plot(stock_df.index, stock_df['average'].values)
plt.show()

# Load and clean data

volume_df = pd.read_csv(cwd + '/volume_2000.csv')
close_df = pd.read_csv(cwd + '/closes_2000.csv', header=0)

close_df.fillna(0, inplace=True)

close_df.index = pd.to_datetime(close_df['Date'])
close_df = close_df.iloc[:, 1:]
'''


def draw_anomaly_plot(scores, prices, test_scores, test_prices, title, lw=2):
    # graphs the calculated anomalies
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)

    ax.plot(np.arange(prices.shape[0]), prices, color='black')
    ax.axhline(0, color='black', lw=2)

    for i in range(0, scores.shape[0]):
        score = scores[i]
        if score < 0:
            l = plt.axvline(x=i, color='red', alpha=0.25, lw=lw)

    ax.plot(np.arange(prices.shape[0], prices.shape[0] + test_prices.shape[0]),
            test_prices, color='green')
    ax.axhline(0, color='black', lw=2)

    for i in range(0, test_scores.shape[0]):
        score = test_scores[i]
        if score < 0:
            l = plt.axvline(x=i+prices.shape[0], color='blue', alpha=0.25, lw=lw)

    plt.show()


def get_iforest(stock_df):
    # stock_df must be cleaned before
    # returns dataframe of classified anomalies
    predict_list = []
    outlier_dict = {}
    for ticker in tqdm(stock_df.columns):

        prices = stock_df[ticker].pct_change().values[1:]
        prices = np.log(prices)
        prices = np.nan_to_num(prices, posinf=0, neginf=0)
        train_pct = 0.75
        cutoff = math.ceil(len(prices) * train_pct)
        train_prices, test_prices = prices[:cutoff], prices[cutoff:]
        clf = IsolationForest(max_samples=100, contamination=0.03).fit(
            train_prices.reshape(-1, 1))
        y_pred_train = clf.predict(train_prices.reshape(-1, 1))
        y_pred_test = clf.predict(test_prices.reshape(-1, 1))

        total_pred = np.concatenate((y_pred_test, y_pred_train), axis=0)
        total_list = list(total_pred)
        predict_list.append(total_list)
        '''
        draw_anomaly_plot(y_pred_train, train_prices,
                          y_pred_test, test_prices, 'title')
        '''
    return pd.DataFrame(predict_list).T


# create an anomaly index
'''
predicts = get_iforest(close_df)
predicts.index = close_df.index[1:]

predicts['anomaly index'] = predicts.sum(axis=1)

predicts.to_csv('anomaly_df.csv')
'''


def triangle(arr):
    N = arr.shape[0]
    li = []
    for j in range(1, N):
        k = j
        while k > 0:
            li.append(arr[j][k-1])
            k -= 1
    return li


def cross_corr(y1, y2):
    """Calculates the cross correlation and lags without normalization.

    The definition of the discrete cross correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html

    Args:
      y1, y2: Should have the same length.

    Returns:
      max_corr: Maximum correlation without normalization.
      lag: The lag in terms of the index.
    """
    if len(y1) != len(y2):
        raise ValueError('The lengths of the inputs should be the same.')

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(
        np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2

    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)
    return max_corr, argmax_corr - shift


vix = pdr.get_data_yahoo('VIXY')
vix_pct = vix['Close'].pct_change()

predicts = pd.read_csv('anomaly_df.csv', index_col=0)
predicts.index = pd.to_datetime(predicts.index)
vals = predicts.iloc[:, -1].pct_change().values
Prophet_df = pd.DataFrame()
Prophet_df['ds'] = predicts.index
Prophet_df['y'] = vals

m = Prophet()
m.fit(Prophet_df)
end = dt.datetime(2017, 3, 6)
future = m.make_future_dataframe(periods=1200)
forecast = m.predict(future)

forecast.index = forecast['ds']

vix_prophet_df = pd.concat([forecast['yhat'], vix_pct], axis=1)
vix_prophet_df.dropna(axis=0, how='any', inplace=True)
vix_prophet_df.columns = ['Anomaly Index Predictions', 'VIXY']

max_corr, lag = cross_corr(vix_prophet_df.iloc[:, 0].values,
                           vix_prophet_df.iloc[:, 1].values)

print(max_corr, lag)
vix_prophet_df.plot()
plt.show()

'''

plt.plot(vix_pct[:forecast['ds'].values[-1]])
plt.plot(forecast[:vix_pct.index.values[0], 'ds'], forecast['yhat'])
plt.show()


fig1 = m.plot(forecast)
plt.show()

corrs = pd.Series(triangle(predicts.iloc[:, :-1].corr().values))
corrs.plot.kde()
plt.show()


plt.plot(predicts.index, predicts['anomaly index'])
plt.show()
'''
