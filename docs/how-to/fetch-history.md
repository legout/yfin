# Fetch history

Use [`history`](../reference/api.md) (sync) or
[`history_async`](../reference/api.md) to pull OHLCV
bars, dividends, and splits from Yahoo's `v8/finance/chart/{symbol}`
endpoint.

## Pick a date range or a period string

You can specify the window with explicit dates **or** a Yahoo range string,
but never both:

```python
import yfin
from datetime import date

# Explicit dates
history = yfin.history(
    ["AAPL", "MSFT"],
    start=date(2024, 1, 1),
    end=date(2024, 6, 1),
)

# Period string
history = yfin.history(["AAPL"], period="1y")

# No window — Yahoo returns everything available
history = yfin.history(["AAPL"])
```

Acceptable `period` values are listed in [Reference / Range](../reference/arrow-schemas.md#range).

## Choose the bar interval

The default is `"1d"`. Other common choices:

```python
history = yfin.history(["AAPL"], period="5d", interval="1h")
```

`Interval` is a `StrEnum` so you can import the valid values rather than
typing the strings:

```python
from yfin import Interval

history = yfin.history(["AAPL"], period="5d", interval=Interval.H1)
```

Yahoo caps some intervals at shorter windows. For example, intraday intervals
(`1m`, `5m`, `15m`, …) only return the most recent ~60 days.

## Include pre/post market

```python
history = yfin.history(
    ["AAPL"],
    period="5d",
    interval="1h",
    include_pre_post=True,
)
```

Pre- and post-market data extends the regular session; the returned rows are
still tagged with the exchange's timezone, not local time.

## Drop the dividend or split columns

By default yfin includes both event columns. Pass `events=()` to skip them
entirely (faster responses, smaller tables):

```python
history = yfin.history(["AAPL"], period="1y", events=("div",))  # splits only
history = yfin.history(["AAPL"], period="1y", events=())  # neither
```

The schema stays the same — the columns are present and just contain nulls.

## Async

```python
async def main():
    history = await yfin.history_async(["AAPL", "MSFT"], period="1y")


asyncio.run(main())
```

`history_async` accepts the same parameters as `history`.

## What happens when some symbols fail

yfin logs the failures and continues with the symbols that succeeded. If
**every** symbol fails, you get a `YahooApiError` summarising the first three
errors.

```python
import logging

logging.basicConfig(level=logging.WARNING)

history = yfin.history(["BAD1", "BAD2", "AAPL"])
# BAD1, BAD2 logged at WARNING; AAPL is in the table.
```

The empty `history` is a table with the full canonical schema and zero rows.

## What's next

- [Reference / HISTORY_SCHEMA](../reference/arrow-schemas.md#history) for the
  full column list.
- [Fetch fundamentals](fetch-fundamentals.md) if you want quarterly valuation
  and statement data alongside prices.
