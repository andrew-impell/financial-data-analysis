'''
FROM THE DATA PRODUCT:
Financial Industry Regulatory Authority
(69,228 datasets)
REFRESHED
3 days ago, on 21 Mar 2020
FREQUENCY
Daily
DESCRIPTION
FINRA/NASDAQ TRF Short Interest: TSLA
VALIDATEi
http://www.finra.org
PERMALINKi
https://www.quandl.com/data/FINRA/FNSQ_TSLA
'''
import quandl
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import numpy as np

def get_short_prop(df):
    df['Proportion'] = (df['ShortVolume'] - df['ShortExemptVolume'])/df['TotalVolume']
    series = df['Proportion']
    return series

tsla_shorts = quandl.get("FINRA/FNSQ_TSLA", authtoken="-h4jm8-epYp2YfshRaBA")

goog_shorts = quandl.get("FINRA/FNSQ_GOOG", authtoken="-h4jm8-epYp2YfshRaBA")

#series = get_short_prop(goog_shorts)

X = np.random.rand(700)

series = 2*X**2 + 1

plt.plot(np.arange(len(X)),series)
plt.show()
'''
model = pm.auto_arima(series, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=10, max_q=10, # maximum p and q
                      m=1,              # frequency of series
                      d=2,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
'''
#print(model.summary())

#model.plot_diagnostics(figsize=(7,5))
#plt.show()

pred = []
for i in range(len(series)):

    model = ARIMA(series,order=(5,1,0))
    model_fit = model.fit(disp=0)
    fc = model_fit.forecast()

    fc_series = fc[i]

    pred.append(fc_series)
#fc_series = model_fit
# Forecast
n_periods = 50
'''
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(series), len(series)+n_periods)

index_of_s = np.arange(0,len(series))

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)
'''
# Plot
plt.plot(series)
plt.plot(pred, color='darkgreen')
'''
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)
'''
plt.title("Goog Short Proportion")
plt.show()








#ax = autocorrelation_plot(series)
#ax.set_xlim((0, 50))
#plt.show()

'''
# fit model
model = ARIMA(tsla_shorts['Proportion'], order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

def get_optimal_params(series,p_range=2,d_range=2,q_range=2):
    error_list = []
    best_config = ''
    for p in range(p_range):
        print(p)
        for d in range(d_range):
            print(d)
            for q in range(q_range):
                print(q)
                print(f"p,d,q = {p},{d},{q}")
                current_config = (p,d,q)
                X = series.values
                size = int(len(X) * 0.66)
                train, test = X[0:size], X[size:len(X)]
                history = [x for x in train]
                predictions = list()
                for t in range(len(test)):
                    model = ARIMA(history, order=(p,d,q))
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                    #print('predicted=%f, expected=%f' % (yhat, obs))
                error = mean_squared_error(test, predictions)
                error_list.append(error)
                if error <= min(error_list):
                    best_config = current_config

    return best_config

#print(get_optimal_params(series,p_range=5,d_range=2,q_range=2))

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(3,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)



print('Test MSE: %.3f' % error)

fc, se, conf = model_fit.forecast(600, alpha=0.05)
fc_series = pd.Series(fc)
lower_series = pd.Series(conf[:, 0])
upper_series = pd.Series(conf[:, 1])
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
'''
