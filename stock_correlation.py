import scipy.stats as sp
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as pdr
from tqdm import tqdm


def buy_stock(
    real_movement,
    signal,
    initial_money=10000,
    max_buy=1,
    max_sell=1,
):
    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """
    starting_money = initial_money
    states_sell = []
    states_buy = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            '''
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement[i])
            )
            '''
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            # print(
            #     'day %d: buy %d units at price %f, total balance %f'
            #     % (i, buy_units, buy_units * real_movement[i], initial_money)
            # )
            states_buy.append(0)
        return initial_money, current_inventory

    for i in range(real_movement.shape[0] - int(0.025 * len(df))):
        state = signal[i]
        if state == 1:
            initial_money, current_inventory = buy(
                i, initial_money, current_inventory
            )
            states_buy.append(i)
        elif state == -1:
            if current_inventory == 0:
                '''
                print('day %d: cannot sell anything, inventory 0' % (i))
                '''
            else:
                if current_inventory > max_sell:
                    sell_units = max_sell
                else:
                    sell_units = current_inventory
                current_inventory -= sell_units
                total_sell = sell_units * real_movement[i]
                initial_money += total_sell
                try:
                    invest = (
                        (real_movement[i] - real_movement[states_buy[-1]])
                        / real_movement[states_buy[-1]]
                    ) * 100
                except:
                    invest = 0
                '''
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                    % (i, sell_units, total_sell, invest, initial_money)
                )
                '''
            states_sell.append(i)

    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    return states_buy, states_sell, total_gains, invest

# states_buy, states_sell, total_gains, invest = buy_stock(df.Close, signals['signal'])


style.use('ggplot')

cwd = os.getcwd()

tickers = pd.read_html(cwd + '/sp500tickers.html')
tickers = list(tickers[0].iloc[:, 0].astype(str).values)
'''
def ts_df(tickers):
    # Get a dataframe of the closing prices from every ticker in the
    # ticker list
    df_list = []
    exception_list = []
    for ticker in tqdm(tickers):
        try:
            ts = pdr.get_data_yahoo(ticker)
            close = ts['Close']
            df_list.append(close)
        except Exception as e:
            print(e)
            print(f'Could not process {ticker}')
            exception_list.append(ticker)
    out_df = pd.concat(df_list, axis=1)
    return out_df


df = ts_df(tickers)

df.to_csv(cwd + '/sp500_closes.csv')

'''

exception_list = ['BRK.B', 'BF.B']
for e in exception_list:
    tickers.remove(e)

tickers.insert(0, 'Date')

df = pd.read_csv(cwd + '/sp500_closes.csv')
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df.columns = tickers
df.set_index('Date', inplace=True)

df.dropna(how='any', axis=1, inplace=True)
'''
df = df.corr()
df.reindex(df.mean().sort_values().index, axis=1)
plt.matshow(df)
plt.show()
'''
# Generate Negative Correlations

HOW_NEG = 50
neg = df.corr()
negs = neg.unstack()
negso = negs.sort_values(kind='quicksort')
negso = negso[negso.values != -1.0].drop_duplicates()
# negcorrelations = negso
negcorrelations = negso[:HOW_NEG]
'''
negpairs = negcorrelations.index.shape[0]

fig, axs = plt.subplots(negpairs, 1, sharex=True)
fig.tight_layout()

i = 0
for first, second in negcorrelations.index:
    axs[i].plot(df.loc[:, [first, second]].index, df.loc[:, [first, second]])
    axs[i].set_title(f'{first}, {second}')
    axs[i].set_xticks(df.loc[:, [first, second]].index[::100])
    axs[i].set_ylabel('Price')
    i += 1
plt.show()


# Generate Positive Correlations

c = df.corr()

s = c.unstack()
so = s.sort_values(kind="quicksort")
HOW = 10
so = so[so.values != 1.0].drop_duplicates()

correlations = so[-HOW:-1]

pairs = correlations.index.shape[0]

fig, axs = plt.subplots(pairs, 1, sharex=True)
fig.tight_layout()

i = 0
for first, second in correlations.index:
    axs[i].plot(df.loc[:, [first, second]].index, df.loc[:, [first, second]])
    axs[i].set_title(f'{first}, {second}')
    axs[i].set_xticks(df.loc[:, [first, second]].index[::100])
    axs[i].set_ylabel('Price')
    i += 1
plt.show()
'''


def lin_regress(t):
    x = np.arange(len(t))
    slope, intercept, r_value, p_value, std_err = sp.linregress(x, t.T.values)
    return slope


def get_signals(ticker, ticker2, short, buy):
    '''
    Get a signals df from a ticker
    '''
    short_cut_off = short
    buy_cut_off = buy

    # count = int(np.ceil(len(df.loc[:, [ticker]]) * 0.1))
    # find which to short and which to buy
    t1 = df.loc[:, [ticker]]
    t2 = df.loc[:, [ticker2]]
    slope1 = lin_regress(t1)
    slope2 = lin_regress(t2)
    if slope1 > slope2:
        short = False
    else:
        short = True

    signals = pd.DataFrame(index=df.loc[:, [ticker]].index)
    signals['signal'] = 0.0
    signals['trend'] = df.loc[:, [ticker]]
    signals['pct_change'] = signals['trend'].pct_change()
    signals['momentum'] = signals['pct_change'].rolling(10).mean() / 10
    if short:
        signals.loc[signals['momentum'] < short_cut_off, 'signal'] = 0
        signals.loc[signals['momentum'] > short_cut_off, 'signal'] = -1
    else:
        signals.loc[signals['momentum'] < buy_cut_off, 'signal'] = 0
        signals.loc[signals['momentum'] > buy_cut_off, 'signal'] = 1
    '''
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
    '''

    return signals['signal']


def opt_buyshort():

    total_gain_list = []
    best_gain = 0
    avg_gain_list = []
    for b in [1]:
        for s in tqdm(np.arange(0, 0.1, 0.001)):
            # for s in tqdm(np.arange(0, 0.001, 0.00001)):

            for first, second in negcorrelations.index:
                Fvals = df.loc[:, [first]].values

                signal = get_signals(first, second, s, b)

                states_buy, states_sell, total_gains, invest = buy_stock(Fvals, signal)

                # temp_gain = float(total_gains[0])
                total_gain_list.append(total_gains)

            avg_gains = np.mean(total_gain_list)

            avg_gain_list.append(avg_gains)
            if best_gain == 0:
                best_gain = avg_gains

            if avg_gains > best_gain:
                best_gain = avg_gains
                best_s = s
                best_b = 1

    return best_s, best_b, avg_gain_list


s, b, av = opt_buyshort()

plt.plot(np.arange(0, 1, 0.01), av)
plt.show()

print(s, b)
