# yfin-client

Compact Python 3.14 Yahoo Finance client returning `pyarrow.Table`, built on
[fastreq](https://github.com/legout/fastreq)'s `curl_cffi` backend with
browser TLS impersonation (the same trick that makes yahooquery reliable).

Published on PyPI as `yfin-client`; imported as `yfin`.

## What it does

- **Batch quotes** via `query1.finance.yahoo.com/v7/finance/quote`
- **Chart history** (OHLCV + dividends + splits) via `query1.finance.yahoo.com/v8/finance/chart/{symbol}`
- Deterministic **pyarrow.Table** output with explicit schemas
- Optional **Polars** conversion via the `polars` extra
- Async and sync APIs; sync wrappers fail clearly inside a running event loop
- Two-strategy Yahoo cookie/crumb authentication (basic + CSRF fallback)
- Per-proxy-route cookie/crumb isolation (no state leakage between proxies)

## What it does NOT do

- No pandas, numpy, yfinance, yahooquery, requests, pendulum, or lxml
- No quote summaries, company fundamentals, symbol search, or options
- No free proxies or bypass/aggressive behavior
- No compatibility layer for the legacy yfin API

## Installation

```bash
pip install yfin-client            # core (pyarrow output)
pip install 'yfin-client[polars]'  # with Polars conversion
```

Requires `fastreq>=3.0.0`, which is resolved automatically from PyPI.

## Usage

```python
import yfin

# Batch quotes
quotes = yfin.quotes(
    ["AAPL", "MSFT"],
    fields=["regularMarketPrice", "regularMarketVolume", "currency"],
)
# pyarrow.Table: symbol, regular_market_price, regular_market_volume, currency

# Historical OHLCV
from datetime import date

history = yfin.history(
    ["AAPL", "MSFT"],
    start=date(2024, 1, 1),
    end=date(2024, 6, 1),
    interval="1d",
)
# pyarrow.Table: symbol, timestamp, open, high, low, close, adjusted_close,
#                volume, dividend, split_ratio, currency, exchange_timezone

# Or use a range instead of dates
history = yfin.history(["AAPL"], period="1y")

# Async equivalents
quotes = await yfin.quotes_async(["AAPL", "MSFT"])
history = await yfin.history_async(["AAPL"], period="1y")

# Optional Polars conversion (requires 'yfin[polars]')
df = yfin.to_polars(history)
```

## Arrow schemas

### History

| Column             | Type                  |
|--------------------|-----------------------|
| `symbol`           | string                |
| `timestamp`        | timestamp(s, UTC)     |
| `open`             | float64               |
| `high`             | float64               |
| `low`              | float64               |
| `close`            | float64               |
| `adjusted_close`   | float64               |
| `volume`           | int64                 |
| `dividend`         | float64               |
| `split_ratio`      | float64               |
| `currency`         | string                |
| `exchange_timezone`| string                |

### Quotes

Column `symbol` (string) followed by each requested Yahoo field converted from
camelCase to snake_case, in caller order. Missing symbols get a null row;
missing field values are null. When `fields=None`, all Yahoo-returned keys are
included as snake_case columns.

## Cookie and crumb authentication

yfin adopts yahooquery's proven session flow, built on fastreq's `curl_cffi`
backend (no yahooquery or curl_cffi dependency at the API level):

1. **Browser impersonation** — every request carries a real browser's TLS
   fingerprint (JA3/JA4), HTTP/2 settings, and headers via curl_cffi's
   `impersonate` feature (a random recent Chrome/Safari/Firefox target per
   client). This is what keeps Yahoo's bot detection from rate-limiting you.
2. **Warmup** — GET `https://finance.yahoo.com` (redirects followed) seeds
   the session cookie jar; most consent walls resolve through the redirect
   chain automatically.
3. **Crumb** — one GET to `query2.finance.yahoo.com/v1/test/getcrumb` per
   route, cached for the session. If it fails, requests proceed crumb-less:
   the v8 chart API works without a crumb.
4. **CSRF fallback** — if the warmup itself fails, an explicit consent flow
   (guce consent form parse, `collectConsent` POST, `copyConsent`) runs once.

Every API request carries yahooquery's default query parameters
(`lang=en-US&region=US&corsDomain=finance.yahoo.com`) plus the crumb when
available. Both quotes (v7) and history (v8/chart) use the `query2` host.

Each network route (direct or a specific proxy URL) maintains independent
cookie/crumb state. A crumb obtained through one proxy is never sent through
another.

## Proxy policy

yfin uses fastreq's explicit proxy transport support. It chooses configured
proxies round-robin as explicit Yahoo routes so that cookie/crumb state remains
bound to the proxy that obtained it. No free proxies are used or offered.
Configure proxies explicitly:

```python
client = yfin.YahooClient(proxies=["http://proxy1:8080", "http://proxy2:8080"])
quotes = await yfin.quotes_async(["AAPL"], client=client)
```

## Unofficial endpoints

yfin uses unofficial Yahoo Finance endpoints. These endpoints are not
documented, not guaranteed stable, and may change without notice. yfin
implements conservative pacing and retry behaviour to minimise the risk of
rate-limiting, but cannot guarantee availability.

## Testing

All tests are hermetic (no network access required):

```bash
uv run pytest -q
```

A live Yahoo smoke test is available but opt-in only (never required for test
success):

```bash
YFIN_LIVE_SMOKE=1 uv run pytest tests/test_live_smoke.py -q
```

## Migration from legacy yfin (pre-1.0)

This is a clean rewrite. The legacy API (`QuoteSummary`, `Search`, `Lookup`,
`validate`, pandas DataFrames) has been removed. Key changes:

- Output is `pyarrow.Table`, not `pandas.DataFrame`
- `history()` uses `interval=` (not `freq=`) and `events=` (not `splits=`/`dividends=`)
- No `QuoteSummary`, symbol search, or lookup endpoints
- No free proxy scraping
- No `pendulum`, `lxml`, `numpy`, `pandas`, `yfinance`, or `parallel-requests`
