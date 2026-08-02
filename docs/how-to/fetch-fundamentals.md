# Fetch fundamentals

Use [`fundamentals`](../reference/api.md) (sync) or
[`fundamentals_async`](../reference/api.md) to pull
dated valuation, income statement, balance sheet, and cash flow series from
Yahoo's `fundamentals-timeseries` endpoint.

## Pick a type group

yfin ships four predefined groups so you don't have to remember Yahoo's
camelCase type names:

```python
import yfin

valuation = yfin.fundamentals(["AAPL", "MSFT"], types=yfin.VALUATION_TYPES)
income = yfin.fundamentals(["AAPL"], types=yfin.INCOME_STATEMENT_TYPES)
balance = yfin.fundamentals(["AAPL"], types=yfin.BALANCE_SHEET_TYPES)
cash = yfin.fundamentals(["AAPL"], types=yfin.CASH_FLOW_TYPES)

# Everything at once
all_fundamentals = yfin.fundamentals(["AAPL"], types=yfin.ALL_TYPES)
```

See [Reference / Module groups](../reference/module-groups.md#fundamentals)
for the full list of types in each group.

## Combine groups, or build a custom list

The groups are plain `list[str]` — concatenate them or compose your own:

```python
custom = (
    yfin.VALUATION_TYPES + ["BasicEPS", "DilutedEPS"]  # add a couple of extras
)
table = yfin.fundamentals(["AAPL"], types=custom)
```

## Pick a date window

The default is the last 4 years (Yahoo ignores earlier `period1` values
anyway). Override with `start` / `end`:

```python
from datetime import date

table = yfin.fundamentals(
    ["AAPL"],
    types=yfin.VALUATION_TYPES,
    start=date(2020, 1, 1),
    end=date(2024, 12, 31),
)
```

`start` / `end` accept `datetime.date`, `datetime.datetime`, or epoch seconds
(integers).

## Read the output

The table has three classes of columns:

- `symbol` (`string`)
- `as_of_date` (`date32`) — the report date
- One column per requested type, converted to snake_case

Numeric types in `INTEGER_TYPES` (`MarketCap`, `EnterpriseValue`,
`BasicAverageShares`, `DilutedAverageShares`) come back as `int64`; all other
types are `float64`.

```python
import polars as pl

df = pl.from_arrow(yfin.fundamentals(["AAPL"], types=yfin.VALUATION_TYPES))

print(
    df.filter(pl.col("as_of_date") >= pl.date(2023, 1, 1))
    .select(["as_of_date", "market_cap", "pe_ratio", "forward_pe_ratio"])
    .sort("as_of_date")
)
```

## What's next

- [Reference / Arrow schemas / Fundamentals](../reference/arrow-schemas.md#fundamentals)
  documents the column types and integer/float split.
- [Fetch summary modules](fetch-summary-modules.md) if you want a current
  snapshot (no date range) of profile, statistics, or ownership data.
