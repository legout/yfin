# Arrow schemas

yfin returns `pyarrow.Table` with deterministic schemas. The schemas are
defined in `yfin.models` and re-exported from the package root for
convenience (`yfin.HISTORY_SCHEMA`).

## Quotes

`yfin.quotes` / `yfin.quotes_async` return a table whose first column is
`symbol` (string), followed by one column per requested Yahoo field converted
to snake_case, in caller order.

When you pass `fields=None` the columns are determined by whatever Yahoo
happened to send back, deduplicated and converted to snake_case.

| Column        | Type                | Notes                                      |
|---------------|---------------------|--------------------------------------------|
| `symbol`      | `string`            | Always first                               |
| `<field>`     | inferred            | One column per Yahoo field, snake_case     |

Missing symbols get a null row. Missing field values for a present symbol
are null.

The Arrow type of each non-symbol column is inferred from the Python values
returned by Yahoo:

- all bool → `bool_`
- all int (and not bool) → `int64`
- all float/int → `float64`
- otherwise → `string`
- all null → `string` (safe default)

## History

`yfin.history` / `yfin.history_async` return a table that matches
`yfin.HISTORY_SCHEMA`:

| Column              | Type                       | Notes                                     |
|---------------------|----------------------------|-------------------------------------------|
| `symbol`            | `string`                   |                                           |
| `timestamp`         | `timestamp(s, tz="UTC")`   | One row per bar                           |
| `open`              | `float64`                  |                                           |
| `high`              | `float64`                  |                                           |
| `low`               | `float64`                  |                                           |
| `close`             | `float64`                  |                                           |
| `adjusted_close`    | `float64`                  |                                           |
| `volume`            | `int64`                    |                                           |
| `dividend`          | `float64`                  | Null unless that day had a dividend       |
| `split_ratio`       | `float64`                  | Null unless that day had a split          |
| `currency`          | `string`                   | From `meta.currency`                      |
| `exchange_timezone` | `string`                   | From `meta.exchangeTimezoneName`          |

`timestamp` is a timezone-aware Arrow timestamp in UTC; use
`pa.compute.localize(...)` to convert to a specific zone.

## Fundamentals

`yfin.fundamentals` / `yfin.fundamentals_async` produce a table with:

| Column       | Type       | Notes                                |
|--------------|------------|--------------------------------------|
| `symbol`     | `string`   |                                      |
| `as_of_date` | `date32`   | Yahoo's `asOfDate` (no time / no tz) |
| `<type>`     | mixed      | One column per requested type        |

Numeric types fall into two buckets (see `yfin.models.INTEGER_TYPES`):

- `int64` — `MarketCap`, `EnterpriseValue`, `BasicAverageShares`,
  `DilutedAverageShares`
- `float64` — everything else

Column names are the Yahoo type names converted to snake_case
(`ForwardPeRatio` → `forward_pe_ratio`).

## Summary

`yfin.quote_summary` and the convenience wrappers produce a table whose
column names are dotted, then converted to snake_case with `.` replaced by
`_`:

| Module kind               | Example column                                | Row shape                |
|---------------------------|-----------------------------------------------|--------------------------|
| Flat (assetProfile, …)    | `asset_profile.industry`                      | one row per symbol       |
| Calendar (calendarEvents) | `calendar_events.earnings_date`               | one row per symbol       |
| Event (upgradeDowngrade…) | `upgrade_downgrade_history.action`            | one row per event        |

The flat rows and event rows are concatenated row-wise. Event rows have
nulls in the flat-only columns and vice versa.

## Interval

`yfin.Interval` is a `StrEnum` enumerating every Yahoo chart interval
yfin accepts:

| Member | Value  |
|--------|--------|
| M1     | `1m`   |
| M2     | `2m`   |
| M5     | `5m`   |
| M15    | `15m`  |
| M30    | `30m`  |
| M60    | `60m`  |
| M90    | `90m`  |
| H1     | `1h`   |
| D1     | `1d`   |
| D5     | `5d`   |
| WK1    | `1wk`  |
| MO1    | `1mo`  |
| MO3    | `3mo`  |

## Range

`yfin.Range` is a `StrEnum` enumerating every Yahoo chart period yfin
accepts:

| Member | Value  |
|--------|--------|
| D1     | `1d`   |
| D5     | `5d`   |
| MO1    | `1mo`  |
| MO3    | `3mo`  |
| MO6    | `6mo`  |
| Y1     | `1y`   |
| Y2     | `2y`   |
| Y5     | `5y`   |
| Y10    | `10y`  |
| YTD    | `ytd`  |
| MAX    | `max`  |
