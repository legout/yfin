# Fetch summary modules

Use [`quote_summary`](../reference/api.md) (or one of the
named convenience wrappers) to pull profile, statistics, calendar, analyst,
and ownership data from Yahoo's `v10/finance/quoteSummary/{symbol}`
endpoint.

## Pick a module group

yfin ships five predefined groups so the camelCase module names stay out of
your code:

```python
import yfin

profile = yfin.asset_profile(["AAPL", "MSFT"])
stats = yfin.summary_detail(["AAPL"])
calendar = yfin.calendar_events(["AAPL"])
analyst = yfin.recommendation_trend(["AAPL"])
ownership = yfin.institution_ownership(["AAPL"])
```

The convenience wrappers (`asset_profile`, `summary_detail`, `key_statistics`,
`financial_data`, `calendar_events`, `upgrade_downgrade_history`,
`recommendation_trend`, `institution_ownership`, `insider_transactions`) are
thin async functions that call `quote_summary_async` with the right module
list. See [Reference / Module groups](../reference/module-groups.md#summary)
for what's in each group.

## Pass arbitrary modules

For anything not covered by a convenience wrapper, call `quote_summary`
directly:

```python
table = yfin.quote_summary(
    ["AAPL"],
    modules=["insiderHolders", "majorHoldersBreakdown"],
)
```

A single string is also accepted:

```python
table = yfin.quote_summary(["AAPL"], modules="assetProfile")
```

## Async

Every wrapper has an `_async` counterpart:

```python
table = await yfin.asset_profile_async(["AAPL"])
table = await yfin.quote_summary_async(["AAPL"], modules=yfin.ALL_SUMMARY_MODULES)
```

## Understand the output shape

Yahoo's quoteSummary response is a dict-of-dicts. yfin flattens each module's
fields into one row per symbol, prefixed by the module name:

```
symbol: string
assetProfile.industry: string
assetProfile.website: string
assetProfile.fullTimeEmployees: int64
summaryDetail.marketCap: int64
defaultKeyStatistics.forwardPE: double
...
```

Some modules (`upgradeDowngradeHistory`, `institutionOwnership`,
`insiderTransactions`, …) produce one row **per event** rather than one row
per symbol. yfin concatenates flat rows and event rows into the same table;
event rows have nulls in the flat-only columns and vice versa.

## All modules at once

```python
all_modules = yfin.quote_summary(["AAPL"], modules=yfin.ALL_SUMMARY_MODULES)
```

This is roughly 50 columns. Prefer a narrower set when you can — Yahoo caps
the response size and drops modules when over-quota.

## What's next

- [Reference / Arrow schemas / Summary](../reference/arrow-schemas.md#summary)
  describes the column prefix conventions and the flat-vs-event split.
- [Explanation / Authentication flow](../explanation/authentication-flow.md)
  explains why this endpoint sometimes needs a CSRF fallback.
