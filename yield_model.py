from tqdm import tqdm
import pickle
from pmdarima.preprocessing import BoxCoxEndogTransformer
from pmdarima.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pmdarima as pm
style.use('ggplot')

# Clean Data

fed_yield = pd.read_csv('feds200628.csv', header=0, skiprows=9)
fed_yield.fillna(0, inplace=True)
fed_yield['Date'] = pd.to_datetime(fed_yield['Date'])
fed_yield.set_index('Date', inplace=True)

yields = fed_yield.iloc[:, 7:37]
modelbeta = fed_yield.iloc[:, :4]
modeltau = fed_yield.iloc[:, -2:]
model = pd.concat([modelbeta, modeltau], axis=1)


def svensson_model(model, n=0):
    '''
    Returns the forward rate based on a time series input
    of Svensson model parameters

    params
    ------
    model: DataFrame
        model parameters DataFrame

    returns
    -------
    total: numpy array
        model output time series

    '''
    beta0 = model.iloc[:, 0].values
    beta1 = model.iloc[:, 1].values
    beta2 = model.iloc[:, 2].values
    beta3 = model.iloc[:, 3].values
    tau1 = model.iloc[:, 4].values
    tau2 = model.iloc[:, 5].values

    one_two = beta0 + beta1 * np.exp(-n / tau1)

    three = beta2 * (n / tau1) * np.exp(-n / tau1)

    four = beta3 * (n / tau2) * np.exp(-n / tau2)

    total = one_two + three + four

    return total


use_N = 1


def plot_shift_model(model, yields, use_N):
    '''
    plots the difference between the model predictions
    and the actual yield

    params
    ------
    model: DataFrame
        The model parameters DataFrame
    yields: DataFrame
        yield time series DataFrame
    use_N: int
        shift value for model

    returns
    -------
    None
    '''
    shift_N = 252 * use_N

    mod = svensson_model(model, use_N)
    diff = \
        mod - \
        yields.iloc[:, 0].shift(periods=shift_N, fill_value=0, axis=0).values

    plt.plot(yields.iloc[:, 0].index, diff)
    plt.show()

    return 0


N_predict = 252


def regress_model_params(model, N_predict):
    '''
    Predicts future values of the svensson model based
    on previous model parameters

    params
    ------
    model: DataFrame
        The model paramter DataFrame
    N_predict: int
        The number of days to predict in the future
    returns
    -------
    pandas_arima: DataFrame

    '''
    arimad = []

    for mcol in tqdm(model.columns):
        pipeline = Pipeline([
            ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
            ('arima', pm.AutoARIMA(seasonal=True, m=12,
                                   suppress_warnings=True,
                                   trace=True))
        ])
        vals = model[mcol].values
        pipeline.fit(vals)

        with open(f'model{mcol}.pkl', 'wb') as pkl:
            pickle.dump(pipeline, pkl)

        # Load model
        with open(f'model{mcol}.pkl', 'rb') as pkl:
            mod = pickle.load(pkl)
            preds = mod.predict(N_predict)

        arimad.append(preds)

    pandas_arima = pd.DataFrame(arimad, columns=model.columns)

    return pandas_arima


pred_model = svensson_model(model)
