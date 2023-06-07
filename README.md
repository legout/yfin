# yFin - A blazing fast python wrapper for the Yahoo Fincance API

Yfin is yet another python wrapper for the Yahoo Finance API. 

By using asyncronous coroutines the http requests are performed in parallel and downloading data is blazing fast. Additionally, to avoid yahoo finance rate limiting, you can use random proxies. 
 
For every function an async version is available. e.g. for downloading hisorical data there is the function `history` and `history_async`. 

This python library currently has functions to download
 - historical OHLCV data
 - fundamental data
 - quotes
 - symbol search

## Installation
```
pip install git+https://github.com/legout/yfin.git
```

## Usage

[Example Notebook](examples/examples.ipynb)

```python
import datetime as dt
import pandas as pd

# Use nest_asyncio, when you are in a jupyter notebook/lab
import nest_asyncio
nest_asyncio.apply()
```
#### 1. Hisorical OHLCV data

Historical data can be downloaded for in several symbols in parallel. The number of symbols is limited to around 1000 if you do not use random proxies.
 
The time range can be set by using the parameters `start` and `end`. Both are timestamps, which can be defined as a string (e.g. `'2022-01-01'` or `'20220101'`) or a `datetime` object (e.g. `datetime.datetime(2022,1,1)`). If `end=None`, today is used as the end timestamp. 
 
Instead of using `start` and  `end`, one can also provide the timerange with the parameter  `period`. Valid optionsare  `1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, 25y, ytd, max`.
 
The parameter `freq` defines the interval between two data points. Valid options are `1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo`.

```python
from yfin.history import history, history_async

# list of symbols can be of any size. Any valid yahoo symbol is supported. Symbols are available for this types: 
# equity, etf, mutualfund, futures, currency, cryptocurrency, index

symbols = ["AAPL", "SF.ST", "DAX", "EUR=X", "BTC-USD", "HOH25.NYM", "0P00000WCP", "^SPX"] 

# sync function. wraps the async function called with asyncio.run
data = history(symbols=symbols, start=dt.datetime(2022,1,1), end=dt.datetime.now(), freq="1d")

# async function
data = await history_async(symbols=symbols, start='20220101', end=None, freq="1d")
```

#### 2. Quote Summary
#
The python class `QuoteSummary` is used to get all the data available on yahoo finance for a specific financial asset. Most of the following properties are only valid for symbols of type equity (stocks).

Following properties are available:
 
 - assetProfile
 - balanceSheetHistory
 - balanceSheetHistoryQuarterly
 - calendarEvents
 - cashflowStatementHistory
 - cashflowStatementHistoryQuarterly
 - defaultKeyStatistics
 - earnings
 - earningsHistory
 - earningsTrend
 - esgScores
 - financialData
 - fundOwnership
 - fundPerformance
 - fundProfile
 - indexTrend
 - incomeStatementHistory
 - incomeStatementHistoryQuarterly
 - industryTrend
 - insiderHolders
 - insiderTransactions
 - institutionOwnership
 - majorHoldersBreakdown
 - pageViews
 - price
 - quoteType
 - recommendationTrend
 - secFilings
 - netSharePurchaseActivity
 - sectorTrend
 - summaryDetail
 - summaryProfile
 - topHoldings
 - upgradeDowngradeHistory

```python
from yfin.quote_summary import QuoteSummary

qs = QuoteSummary(symbols=["AAPL", "NET", "VOW.DE", "SF.ST"])

# use `fetch` to get the data for the given modules.
res = await qs.fetch(modules=["upgradeDowngradeHistory", "insider_holders"]) # you can use camel or snake case for the property names

# Use class properties 
# Summary Profile
pd.DataFrame(qs.summary_profile)

# Default Key Stats
pd.concat(qs.default_key_statistics).unstack()

# Income Statement
pd.concat(qs.income_statement_history)

# async function for earnings_history 
pd.concat(await qs.earnings_history_async())
```

#### 3. Symbol Search

**Classic symbol search**

 Yahoo fincane offers two different symbol search endpoints which are available in yFin in the the function `search` (or its async equivalent `search_async`).
 
By using the parameter `search_assist` one can switch between the two endpoints, whereas `search_assist=1` uses the endpoint, which is used in the search bar on [finance.yahoo.com](finance.yahoo.com).
The search query can be an asset name, or its symbol. Results are limited to 6 and 10 for `search_assist=1` and `search_assist=2`, respectively.

```python
from yfin.symbols import search

search("Volkswagen", search_assist=1) # search_assist=1 is the default
search("Volkswagen", search_assist=2)
```

**Lookup symbol search**

Additionally, a lookup endpoint is included in yFin, which can also be used for searching symbols. This endpoints gives up to 10000 results for a search query, but is less accurate than the classic symbol search functions. It is also possible to search for (a) specific asset type(s) and most important, it seems that there isn´t any rate limiting for this endpoint. 
 
Therefore this function is especially useful to get (nearly) all available symbols for (a) specific asset type(s). :-)  This is exactly what I am doing in my other yahoo finance related python module [yahoo symbols](https://github.com/legout/yahoo-symbols).

```python
from yfin.symbols import lookup_search

lookup_search("btc", type_="cryptocurrency")
```


#

## Use of a random proxy server.

**Note** The library should work fine without using random proxies, but the number of requests is limited to ~1000/h. Using random proxies might be illegal.

You can set `random-proxy=True` in any of the yFin functions. By default this uses free proxies*. In my experience, these proxies are not reliable, but maybe you are lucky.

It is also possible to provide a list or `proxies` as a parameter in every yFin function. 

```python
from yfin import history
from yfin.utils.proxy import get_free_proxy_list

data = history("AAPL", random_proxy=True)

proxies = ["http://proxy.one.com", "https://proxy.town.com"]
# or get free proxies. Note: These proxies are not very reliable
# proxies = get_free_proxy_list()

data = history("AAPL", proxies=proxies)
```

### Webshare.io proxies

I am using proxies from [webshare.io](https://www.webshare.io/). I am very happy with their service and the pricing. If you wanna use their service too, sign up (use the [this link](https://www.webshare.io/?referral_code=upb7xtsy39kl) if you wanna support my work) and choose a plan that fits your needs. In the next step, go to Dashboard -> Proxy -> List -> Download and copy the download link. Set this download link as an environment variable `WEBSHARE_PROXIES_URL`  before importing any yFin function. 

*Export WEBSHARE_PROXIES_URL in your linux shell*
```
$ export WEBSHARE_PROXIES_URL="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/"
```

You can also set this environment variable permanently in an `.env` file (see the `.env-exmaple`) in your home folder or current folder or in your command line config file (e.g. `~/.bashrc`).

*Write WEBSHARE_PROXIES_URL into .env*
```
WEBSHARE_PROXIES_URL="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/"
```

*or write WEBSHARE_PROXIES_URL into your shell config file (e.g. ~/.bashrc)*
```
$ echo 'export WEBSHARE_PROXIES_URL="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/"' >> ~/.bashrc
```

The last option is to set your `WEBSHARE_PROXIES_URL` within your python code. **Note** It is neccessary to do before importing any other yFin function.

```python
from yfin.utils.proxy import set_webshare_proxies_url

set_webshare_proxies_url(url="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/")
```



*Free Proxies are scraped from here:
- "http://www.free-proxy-list.net"
- "https://free-proxy-list.net/anonymous-proxy.html"
- "https://www.us-proxy.org/"
- "https://free-proxy-list.net/uk-proxy.html"
- "https://www.sslproxies.org/"


<hr>

#### Support my work :-)

If you find this useful, you can buy me a coffee. Thanks!

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/W7W0ACJPB)

