# Module groups

yfin ships a few curated lists of Yahoo's camelCase identifiers so you don't
have to memorise them. They're plain `list[str]` exported from the package
root, so you can use them directly or compose your own.

## Fundamentals

Use with `yfin.fundamentals(...)`.

### `VALUATION_TYPES`

Valuation ratios and market-cap sized fields:

- `ForwardPeRatio`
- `PsRatio`
- `PbRatio`
- `EnterprisesValueEBITDARatio`
- `EnterprisesValueRevenueRatio`
- `PeRatio`
- `MarketCap`
- `EnterpriseValue`
- `PegRatio`

### `INCOME_STATEMENT_TYPES`

- `TotalRevenue`
- `CostOfRevenue`
- `GrossProfit`
- `OperatingIncome`
- `NetIncome`
- `EBIT`
- `EBITDA`
- `BasicEPS`
- `DilutedEPS`
- `ResearchAndDevelopment`
- `SellingGeneralAndAdministration`
- `InterestExpense`
- `TaxProvision`
- `DilutedAverageShares`
- `BasicAverageShares`
- `OperatingExpense`
- `TotalExpenses`
- `PretaxIncome`
- `NormalizedEBITDA`

### `BALANCE_SHEET_TYPES`

- `TotalAssets`
- `StockholdersEquity`
- `TotalDebt`
- `LongTermDebt`
- `CurrentDebt`
- `CashAndCashEquivalents`
- `Inventory`
- `Goodwill`
- `NetPPE`
- `WorkingCapital`
- `RetainedEarnings`
- `CurrentAssets`
- `CurrentLiabilities`
- `NetDebt`
- `CommonStockEquity`
- `TangibleBookValue`

### `CASH_FLOW_TYPES`

- `OperatingCashFlow`
- `FreeCashFlow`
- `CapitalExpenditure`
- `RepurchaseOfCapitalStock`
- `CashDividendsPaid`
- `NetCommonStockIssuance`
- `Depreciation`
- `StockBasedCompensation`
- `EndCashPosition`
- `NetIncomeFromContinuingOperations`

### `ALL_TYPES`

Concatenation of the four groups above.

## Summary

Use with `yfin.quote_summary(...)` or one of the convenience wrappers.

### `PROFILE_MODULES`

`["assetProfile", "quoteType"]`

Convenience wrapper: [`yfin.asset_profile`](../reference/api.md).

### `STATS_MODULES`

`["summaryDetail", "defaultKeyStatistics", "financialData"]`

Convenience wrapper: [`yfin.summary_detail`](../reference/api.md).

### `CALENDAR_MODULES`

`["calendarEvents"]`

Convenience wrapper: [`yfin.calendar_events`](../reference/api.md).

### `ANALYST_MODULES`

`["upgradeDowngradeHistory", "recommendationTrend", "earningsHistory"]`

Convenience wrappers: [`yfin.upgrade_downgrade_history`](../reference/api.md),
[`yfin.recommendation_trend`](../reference/api.md),
[`yfin.earnings_history`](../reference/api.md).

### `OWNERSHIP_MODULES`

`["institutionOwnership", "fundOwnership", "majorHoldersBreakdown",
"insiderHolders", "insiderTransactions"]`

Convenience wrappers:
[`yfin.institution_ownership`](../reference/api.md),
[`yfin.fund_ownership`](../reference/api.md),
[`yfin.major_holders_breakdown`](../reference/api.md),
[`yfin.insider_holders`](../reference/api.md),
[`yfin.insider_transactions`](../reference/api.md).

### `INSIDER_TRANSACTIONS_MODULES`

`["insiderTransactions"]`

Dedicated group for the insider-transactions array module, parallel to
`CALENDAR_MODULES`. Same single module as the `insiderTransactions` member
of `OWNERSHIP_MODULES`, but usable on its own. Convenience wrapper:
[`yfin.insider_transactions`](../reference/api.md).

### `TREND_MODULES`

`["indexTrend", "sectorTrend", "pageViews"]`

Index/sector valuation trends and page-view momentum. No dedicated
wrapper — request via `quote_summary`.

### `ALL_SUMMARY_MODULES`

Concatenation of the groups above (insiderTransactions appears once, via
`OWNERSHIP_MODULES`).

### `SUMMARY_MODULES`

Sorted registry of **every** quoteSummary module yfin can materialise
(24 modules). `quote_summary` validates its `modules` argument against
this list and raises `ValueError` for unknown names — Yahoo would
otherwise silently drop them. Modules not covered by a curated group or
wrapper (e.g. `price`, `summaryProfile`, `earnings`,
`netSharePurchaseActivity`) are requested via `quote_summary` directly:

```python
table = yfin.quote_summary(["AAPL"], modules="netSharePurchaseActivity")
```

## Column prefixes and `strip_prefix`

Summary columns are prefixed with the module name
(`insider_transactions_shares`) so multi-module tables never collide.
Single-module convenience wrappers strip the prefix by default
(`shares`); pass `strip_prefix=False` to keep it. Multi-module wrappers
(`asset_profile`, `summary_detail`) keep prefixes because their modules
share field names (e.g. `maxAge`). `quote_summary(..., strip_prefix=True)`
raises `ValueError` when stripping would merge distinct columns.
