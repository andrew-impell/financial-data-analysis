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
file: short_interest_tsla.py

Use time series analysis to predict optimal exit and entry of short positions using FINRA/NASDAQ TRF Short Interest data from [Quandl](https://quandl.com).

# Svensson Treasury Yield Model
file: yield_model.py

Regreeses Federal Reserve model parameters to predict futures values. Data can be found [here](https://www.federalreserve.gov/data/nominal-yield-curve.htm).

# Market News Sentiment
file: sentiment_news.py

helper files: process_sentiment.py, get_news_sites.py


Uses `newspaper` to grab headlines and uses the `flair` NLP library for sentiment analysis to find the overall positivity or negativity of the news. Will attempt to integrate it into trading strategies and asset pricing models.



