import quandl
import numpy as np
from stochastic import brownian
import matplotlib.pyplot as plt
import pyflux as pf
import pandas as pd
import statsmodels.api as sm
import datetime


'''
    _arma_model = sm.tsa.arima_model.ARMA(input_ts, (1,1))
    _model_result = _arma_model.fit()
    arch_model(_model_result.resid, mean='Zero', p=1, q=1)

'''

forex = quandl.get("FED/RXI_US_N_B_EU")

forex.index = pd.to_datetime(forex.index, format="%d/%m/%Y")


returns = pd.DataFrame(np.diff(np.log(forex['Value'].values)))
returns.index = forex.index.values[1:forex.index.values.shape[0]]
returns.columns = ['Returns']


forex_dates = forex.index.values


start_date = datetime.datetime(1999, 1, 4)
end_date = datetime.datetime(2020, 4, 3)

difference = end_date - start_date
difference_in_years = (difference.days + difference.seconds/86400)/365.2425

print(difference_in_years)

S0 = forex['Value'].iloc[0]
ST = forex['Value'].iloc[-1]

mu = np.log(ST/S0)/difference_in_years

N_trade = 252
N = 250

normal = np.sqrt(N_trade/N)


sum = 0

n = forex_dates.shape[0]


const = normal/np.sqrt(n-1)

for i in range(n-N):
    sum += np.log(forex['Value'].iloc[i+N]/forex['Value'].iloc[i])**2

var = np.sqrt(sum)*const

process = brownian(S0, n, 1, var**2)
plt.plot(list(forex['Value'].values), label='Actual')
plt.plot(process, label='Predicted')
plt.xlabel('t')
plt.ylabel('FX Rate')
plt.legend()
plt.show()


'''

arma_model = sm.tsa.ARMA(returns, (1, 1))
model_result = arma_model.fit()

vals = model_result.resid.values


ts_df = pd.DataFrame(vals, index=returns.index)


GARCH_model = pf.GARCH(ts_df, p=1, q=1)
x = GARCH_model.fit()
x.summary()

GARCH_model.plot_predict(h=10)
'''
