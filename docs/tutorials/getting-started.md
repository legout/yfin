# Getting started

This tutorial walks through installing yfin, fetching your first batch of
quotes, and inspecting the result. By the end you'll have run four real
network calls against Yahoo Finance and understand what yfin returns.

## What you'll need

- Python 3.14 or later (`python --version` should report `3.14` or higher)
- A virtual environment manager — we'll use `uv`, but `python -m venv` works
  just as well
- Outbound network access to `query2.finance.yahoo.com`

## 1. Set up the environment

```bash
mkdir yfin-sandbox
cd yfin-sandbox
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install 'yfin-client[polars]'
```

The `[polars]` extra is optional — it adds Polars as a conversion target for
the `pyarrow.Table` output. Skip it if you only need Arrow.

## 2. Fetch your first batch of quotes

Create `first_quotes.py`:

```python
import yfin

quotes = yfin.quotes(
    ["AAPL", "MSFT", "GOOGL"],
    fields=["regularMarketPrice", "regularMarketVolume", "currency"],
)

print(quotes.schema)
print(quotes.to_pandas())
```

Run it:

```bash
python first_quotes.py
```

You'll see something like:

```
symbol: string
regular_market_price: double
regular_market_volume: int64
currency: string
  symbol  regular_market_price  regular_market_volume currency
0   AAPL              192.42            48_123_456.0      USD
1   MSFT              415.67            22_987_654.0      USD
2  GOOGL              172.13            25_345_678.0      USD
```

What happened:

- yfin downloaded the session cookie and crumb from Yahoo
- It split the symbols into URL-safe chunks (each request is one comma-separated
  `symbols=...` query parameter)
- It returned a `pyarrow.Table` with one row per symbol, columns in the order
  you asked for

## 3. Pull historical OHLCV

Create `first_history.py`:

```python
import yfin
from datetime import date

history = yfin.history(
    ["AAPL", "MSFT"],
    start=date(2024, 1, 1),
    end=date(2024, 6, 1),
    interval="1d",
)

print(history.num_rows, "rows")
print(history.schema)
print(history.to_pandas().head())
```

Each symbol issues its own request to Yahoo's chart endpoint, and the per-symbol
tables are concatenated row-wise into a single table.

## 4. Ask for fundamentals

Create `first_fundamentals.py`:

```python
import yfin

# Valuation ratios + market cap, last 4 years by default
fundamentals = yfin.fundamentals(["AAPL"], types=yfin.VALUATION_TYPES)

print(fundamentals.schema)
print(fundamentals.to_pandas().tail())
```

The result has one row per `(symbol, as_of_date)` pair and one column per
requested type. Numeric types in `INTEGER_TYPES` (MarketCap, EnterpriseValue,
the share counts) come back as `int64`; everything else is `float64`.

## 5. Switch to Polars

If you installed the `polars` extra, swap the conversion:

```python
import yfin

quotes = yfin.quotes(["AAPL", "MSFT"])
df = yfin.to_polars(quotes)

print(type(df))  # polars.dataframe.frame.DataFrame
print(df)
```

`to_polars` is a thin wrapper over `polars.from_arrow` — it preserves the
exact Arrow schema and only changes the surface API.

## Where next?

- The [How-to guides](../how-to/index.md) cover individual tasks (one call per
  guide) for when you already know what you want.
- [Reference / Arrow schemas](../reference/arrow-schemas.md) documents every
  column and type yfin can produce.
- [Explanation / Architecture](../explanation/architecture.md) explains how the
  client is wired together.
