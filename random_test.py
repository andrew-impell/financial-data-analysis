import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import os

'''
Analyse log returns with NIST's SP-800-22 RNG tests.
NIST library from https://github.com/dj-on-github/sp800_22_tests
'''


working_dir = '/home/andrew/Documents/dev/econ/'

etfs = glob.glob(working_dir + 'ETFs/*')
stocks = glob.glob(working_dir + 'Stocks/*')


stock_list = []
etf_list = []
for etf in tqdm(etfs):
    t_df = pd.read_csv(etf)
    close = t_df['Close']
    etf_list.append(close)

etf_df = pd.concat(etf_list, axis=1)

log_etf = (np.log(etf_df) - np.log(etf_df.shift(1))) + 1
log_etf.fillna(0, inplace=True)


for stock in tqdm(stocks):
    try:
        tdf = pd.read_csv(stock)
        close = tdf['Close']
        stock_list.append(close)
    except Exception as e:
        print(e)

stock_df = pd.concat(stock_list, axis=1)

log_stock = (np.log(stock_df) - np.log(stock_df.shift(1))) + 1
log_stock.fillna(0, inplace=True)

flat_etf = log_etf.values.flatten()
flat_etf.tofile('log_etf')

gc.collect()


flat_stock = log_stock.values.flatten()
flat_stock.tofile('log_stock')

# change dir for your system
# running all tests scripts fails so just run all individually

dir = '~/Documents/dev/econ/market_random/sp800_22_tests/*.py'
binary_file = '~/Documents/dev/econ/market_random/sp800_22_tests/log_etf'

sh_command = \
    f'log={binary_file}; for f in {dir} ; do echo "$f" && "$f$log" ; done'

os.system(sh_command)
