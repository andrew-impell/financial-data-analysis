import quandl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
plt.style.use('ggplot')

MAX_LAG = 30

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def get_cross_list(x,y,MAX_LAG=30):
    """
    Gets a list of cross correlations for each number
    of lagged days
    Parameters
    ----------
    MAX_LAG : int, default 30
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    corr_list : list
    """
    corr_list = []
    for i in range(MAX_LAG+1):
        corr_list.append(crosscorr(x,y,MAX_LAG))
    if all_same(corr_list):
        return crosscorr(x,y)

def all_same(items):
    return all(x == items[0] for x in items)

bond_AAA_yield = quandl.get("ML/AAAEY", authtoken="-h4jm8-epYp2YfshRaBA")
bond_AA_yield = quandl.get("ML/AAY", authtoken="-h4jm8-epYp2YfshRaBA")
bond_A_yield = quandl.get("ML/AEY", authtoken="-h4jm8-epYp2YfshRaBA")
bond_BBB_yield = quandl.get("ML/BBBEY", authtoken="-h4jm8-epYp2YfshRaBA")
bond_BB_yield = quandl.get("ML/BBY", authtoken="-h4jm8-epYp2YfshRaBA")
bond_B_yield = quandl.get("ML/BEY", authtoken="-h4jm8-epYp2YfshRaBA")
bond_CCC_yield = quandl.get("ML/CCCY", authtoken="-h4jm8-epYp2YfshRaBA")

sixm_tres = quandl.get("FRED/T6MFFM", authtoken="-h4jm8-epYp2YfshRaBA",start_date="1996-12-31", end_date="2020-03-19")


all_tres_yields = quandl.get("USTREASURY/YIELD", authtoken="-h4jm8-epYp2YfshRaBA",start_date="1996-12-31", end_date="2020-03-19")

print("---FINISHED LOADING DATA---")

all_tres_yields = all_tres_yields.fillna(0)

tres_yield_dict = dict()

for col in all_tres_yields.columns.values:
    tres_yield_dict[col] = all_tres_yields[col]



bond_yields = {'AAA':bond_AAA_yield,
                'AA': bond_AA_yield,
                'A': bond_A_yield,
                'BBB': bond_BBB_yield,
                'BB': bond_BB_yield,
                'B': bond_B_yield,
                'CCC': bond_CCC_yield}

print('---FINISHED DATA PROCESSING---')

#x_corrs = np.arange(1,32)

#num_plots = len(bond_yields)

#fig, axs = plt.subplots(num_plots, sharex=True)

def plot_asset_correlation(asset_list_1,asset_list_2):

    num_plots_x = len(asset_list_1)
    num_plots_y = len(asset_list_2)

    fig, axs = plt.subplots(num_plots_x,num_plots_y)

    for asset_list_1key, asset_list_1value in asset_list_1.items():

        bar_corrs = dict()

        for asset_list_2key, asset_list_2value in asset_list_2.items():




            #calculate correlation
            col = asset_list_2value.columns[0]
            corrs = crosscorr(asset_list_1value,asset_list_2value[col])


            bar_corrs[asset_list_2key] = corrs

            axes_x = list(asset_list_1.keys()).index(asset_list_1key)
            axes_y = list(asset_list_2.keys()).index(asset_list_2key)

            axs[axes_x,axes_y].bar(range(len(bar_corrs)), list(bar_corrs.values()), align='center')
            axs[axes_x,axes_y].set_xticks(range(len(bar_corrs)))
            axs[axes_x,axes_y].set_xticklabels(list(bar_corrs.keys()))

        plt.show()

    return 0

plot_asset_correlation(sixm_tres,bond_yields)

'''
#Print all yields
for bond_rating,bond_series in bond_yields.items():
    col = bond_series.columns[0]
    plt.plot(bond_series[col],label=bond_rating)

plt.ylabel('Percent Yield')
plt.xlabel('Date')
plt.legend()
plt.title('Yields on Corporate Bond Ratings')
plt.show()
'''
