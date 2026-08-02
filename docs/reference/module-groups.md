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

`["upgradeDowngradeHistory", "recommendationTrend", "earningHistory"]`

Convenience wrappers: [`yfin.upgrade_downgrade_history`](../reference/api.md),
[`yfin.recommendation_trend`](../reference/api.md).

### `OWNERSHIP_MODULES`

`["institutionOwnership", "fundOwnership", "majorHoldersBreakdown",
"insiderHolders", "insiderTransactions"]`

Convenience wrappers:
[`yfin.institution_ownership`](../reference/api.md),
[`yfin.insider_transactions`](../reference/api.md).

### `ALL_SUMMARY_MODULES`

Concatenation of all five groups above. Roughly 50 columns.
