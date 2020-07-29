# financial-data-analysis
Random array of scripts to price securities, analyse market data, etc..

# Correlation
file: stock_correlation.py

Find correlations between pairs and graph the highest.

![](https://github.com/andrew-impell/financial-data-analysis/blob/master/correlations.png)

# Hedge Fund Portfolio

Use SEC 13F form data to create a portfolio of securities traded by hedge funds/institutional traders and then benchmark its returns against the S&P 500. Securities held more often by multiple funds are weighted higher.

![](https://github.com/andrew-impell/financial-data-analysis/blob/master/hedge_fund_picks.png)

Data from [Fintel](https://fintel.io)

# Short Interest
file: short_interest_tsla.pySP800-22 Rev 1a PRNG test suite

Use time series analysis to predict optimal exit and entry of short positions using FINRA/NASDAQ TRF Short Interest data from [Quandl](https://quandl.com).

# Svensson Treasury Yield Model
file: yield_model.py

Regreeses Federal Reserve model parameters to predict futures values. Data can be found [here](https://www.federalreserve.gov/data/nominal-yield-curve.htm).
anomalies
# Market News Sentiment
file: sentiment_news.py

helper files: process_sentiment.py, get_news_sites.py


Uses `newspaper` to grab headlines and uses the `flair` NLP library for sentiment analysis to find the overall positivity or negativity of the news. Will attempt to integrate it into trading strategies and asset pricing models.

# Anomaly Detection with Isolation Forest

Looks at the Kaggle stock dataset to find patterns in the aggregate levels of anomalies over time. `prophet` is used to try to find a time series trend.

![](https://github.com/andrew-impell/financial-data-analysis/blob/master/anom.png)
 <p>Graph of the log(daily percent change) of a single security with in-sample outliers in red and out-of-sample outliers in green.</p>

![](https://github.com/andrew-impell/financial-data-analysis/blob/master/index2.png)
<p>Total of all outliers over time.</p>

![](https://github.com/andrew-impell/financial-data-analysis/blob/master/prophet2.png)

 <p> prophet model on the anomaly index.</p>
 
 # NIST SP800-22 Rev 1a PRNG test suite analysis
 file: random_test.py
 
 
 Uses PRNG tests on log returns to try to test the randomness of market returns.
 Requires [this](https://github.com/dj-on-github/sp800_22_tests) great repository for the NIST tests.
 
 <b>Sample output:</b>
 
 ```
 /home/andrew/Documents/dev/econ/market_random/sp800_22_tests/sp800_22_random_excursion_test.py
J=3
x = -4	chisq = 0.428871	p = 0.994500 
x = -3	chisq = 0.600144	p = 0.987997 
x = -2	chisq = 1.000300	p = 0.962542 
x = -1	chisq = 0.999700	p = 0.962590 
x = 1	chisq = 4.333033	p = 0.502529 
x = 2	chisq = 8.247775	p = 0.143099 
x = 3	chisq = 0.600144	p = 0.987997 
x = 4	chisq = 0.428871	p = 0.994500 
J too small (J < 500) for result to be reliable
success = True
plist =  [0.9945000204954214, 0.9879969685847894, 0.9625415766568545, 0.9625899708254657, 0.5025287424472648, 0.14309859788329615, 0.9879969685847894, 0.9945000204954214]

/home/andrew/Documents/dev/econ/market_random/sp800_22_tests/sp800_22_serial_test.py
  psi_sq_m   =  2.8
  psi_sq_mm1 =  1.2
  psi_sq_mm2 =  0.4
  delta1     =  1.6
  delta2     =  0.8
  P1         =  0.808792135411
  P2         =  0.670320046036
success = True
plist =  [0.8087921354109985, 0.6703200460356384]

/home/andrew/Documents/dev/econ/market_random/sp800_22_tests/sp800_22_maurers_universal_test.py
  sum = 7.16992500144
  fn = 1.19498750024
success = True
p       =  0.0314262987784

etc...

 ```

# Vector Autoregression on Tech Stocks

main file: sp500_tech.py

Creates a VAR model of 11 different tech stocks and predicts future price movement.
