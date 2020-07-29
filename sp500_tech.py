from statsmodels.stats.stattools import durbin_watson
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import style

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split

style.use('ggplot')

'''
with open('sp_tech.html', 'r') as f:
    symbols_html = f.read()

content = BeautifulSoup(symbols_html, features="lxml")

tech_symbols = []

for div in content.find_all("div", {"class": "filterable-list-cell font-bold-style"}):
    for sub_div in div.find_all('span'):
        tech_symbols.append(sub_div.string)

tech_symbols = tech_symbols[:10]

start = datetime(2019, 6, 1)
df_list = []

for ticker in tech_symbols:
    print(ticker)
    df_list.append(pdr.get_data_yahoo(ticker, start=start)['Close'])

df = pd.concat(df_list, axis=1)
df.columns = tech_symbols

df.to_csv('sp500_tech.csv')

'''
df = pd.read_csv('sp500_tech.csv', index_col=0)
df.sort_index(axis=0, inplace=True)

dateframe_length = df.index.values.shape[0]
train_amount = 0.8
idx_cuttoff = int(np.floor(train_amount * dateframe_length))

last_obs = dateframe_length - idx_cuttoff

train_df = df.iloc[:idx_cuttoff]
test_df = df.iloc[idx_cuttoff:]

train_diff = train_df.diff().dropna()
# train_2diff = train_diff.diff().dropna() unecessary


def test_adfuller(series, siginif=0.05):
    r = adfuller(series, autolag='AIC')
    pvalue = r[1]
    output = {'test_statistic': round(r[0], 4),
              'pvalue': round(r[1], 4),
              'n_lags': round(r[2], 4),
              'n_obs': r[3]}
    if pvalue <= siginif:
        print('Stationary\n')
    else:
        print('Non-Stationary\n')

#
# for col in train_df.columns:
#     test_adfuller(train_diff[col])


model = VAR(train_diff)
'''
aic_list = []
bic_list = []

for i in range(1, 11):
    result = model.fit(i)
    print(f'Lag: {i}')
    print(f'AIC: {result.aic}')
    print(f'BIC: {result.bic}')

    aic = result.aic
    bic = result.bic
    aic_list.append(aic)
    bic_list.append(bic)
'''
# print(min(bic_list))
# plt.plot(range(1, 11), aic_list, label='aic')
# plt.plot(range(1, 11), bic_list, label='bic')
#
# plt.legend()
# plt.show()

x = model.select_order(maxlags=19)
print(x.summary())


model_fitted = model.fit(3)

out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(col, ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar


# Input data for forecasting
forecast_input = train_diff.values[-lag_order:]
fc = model_fitted.forecast(y=forecast_input, steps=int(last_obs))
df_forecast = pd.DataFrame(fc, index=df.index[-last_obs:], columns=df.columns + '_2d')


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + \
                df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


df_results = invert_transformation(train_df, df_forecast, second_diff=True)
print(df_results)


fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10, 10))
for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x', tight=True)
    test_df[col][-last_obs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()
