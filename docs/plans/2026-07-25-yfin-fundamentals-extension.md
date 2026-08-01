# yfin Extension: Fundamentals + QuoteSummary

> **For Hermes:** Implement module by module with TDD. Each module is independent.

## Goal

Extend yfin with two new endpoint families:
1. **`fundamentals.py`** — `fundamentals-timeseries` (ws endpoint): dated valuation ratios, financial statements, shares outstanding (~4yr history)
2. **`summary.py`** — `quoteSummary` (v10 endpoint): company profiles, key statistics, earnings, analyst grades, ownership data

Both return deterministic `pyarrow.Table` output, consistent with existing `history_async` / `quotes_async`.

---

## Endpoint reference

### fundamentals-timeseries
```
GET https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}
  ?period1={epoch_start}
  &period2={epoch_end}
  &type={comma_separated_fields}
  &merge=false
  &padTimeSeries=false
```

Response shape:
```json
{
  "timeseries": {
    "result": [{
      "meta": {"symbol": "AAPL", ...},
      "timestamp": [1718841600, ...],
      "PeRatio": [{"asOfDate": "2024-06-10", "reportedValue": {"raw": 30.03}}, ...]
    }]
  }
}
```

Server-side cap: ignores `period1` earlier than ~4 years. Each `type` field returns its own time-series with `asOfDate` stamps.

### quoteSummary
```
GET https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}
  ?modules={comma_separated_modules}
```

Response shape:
```json
{
  "quoteSummary": {
    "result": [{
      "assetProfile": {...},
      "summaryDetail": {...},
      ...
    }]
  }
}
```

No date range — returns current snapshot (or rolling multi-period for statement modules).

---

## Module 1: `fundamentals.py`

### Predefined type groups

```python
VALUATION_TYPES = [
    "ForwardPeRatio",
    "PsRatio",
    "PbRatio",
    "EnterprisesValueEBITDARatio",
    "EnterprisesValueRevenueRatio",
    "PeRatio",
    "MarketCap",
    "EnterpriseValue",
    "PegRatio",
]

INCOME_STATEMENT_TYPES = [
    "TotalRevenue",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncome",
    "NetIncome",
    "EBIT",
    "EBITDA",
    "BasicEPS",
    "DilutedEPS",
    "ResearchAndDevelopment",
    "SellingGeneralAndAdministration",
    "InterestExpense",
    "TaxProvision",
    "DilutedAverageShares",
    "BasicAverageShares",
    "OperatingExpense",
    "TotalExpenses",
    "PretaxIncome",
    "NormalizedEBITDA",
]

BALANCE_SHEET_TYPES = [
    "TotalAssets",
    "TotalLiabilitiesNetMinorityInterest",
    "StockholdersEquity",
    "TotalDebt",
    "LongTermDebt",
    "CurrentDebt",
    "CashAndCashEquivalents",
    "Inventory",
    "Goodwill",
    "NetPPE",
    "WorkingCapital",
    "RetainedEarnings",
    "CurrentAssets",
    "CurrentLiabilities",
    "NetDebt",
    "CommonStockEquity",
    "TangibleBookValue",
]

CASH_FLOW_TYPES = [
    "OperatingCashFlow",
    "FreeCashFlow",
    "CapitalExpenditure",
    "RepurchaseOfCapitalStock",
    "CashDividendsPaid",
    "NetCommonStockIssuance",
    "Depreciation",
    "StockBasedCompensation",
    "EndCashPosition",
    "NetIncomeFromContinuingOperations",
    "LongTermDebtPayments",
    "IssuanceOfDebt",
]
```

### API

```python
async def fundamentals_async(
    symbols: str | list[str],
    *,
    types: list[str],
    start: date | None = None,
    end: date | None = None,
    client: YahooClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch dated fundamental time-series data.

    Returns Arrow table with columns:
      symbol (utf8), asOfDate (date32), periodType (utf8),
      then one column per requested type (float64 or int64).
    """


def fundamentals(
    symbols: str | list[str],
    *,
    types: list[str],
    start: date | None = None,
    end: date | None = None,
    client: YahooClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper."""
```

### Arrow schema

```python
FUNDAMENTALS_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("asOfDate", pa.date32()),
        # dynamic: one float64 field per requested type
        # special: MarketCap, EnterpriseValue, BasicAverageShares → int64
    ]
)
```

### Implementation notes

- 1 symbol per request (endpoint is per-symbol)
- Use `return_exceptions=True` in gather (same pattern as history_async fix)
- Parse `timestamp` array + per-type `asOfDate` arrays
- Merge into wide table: one row per `(symbol, asOfDate)`, columns per type
- Server ignores early `period1` — no client-side date filtering needed

---

## Module 2: `summary.py`

### Predefined module sets

```python
PROFILE_MODULES = ["assetProfile", "quoteType", "summaryProfile"]
STATS_MODULES = ["summaryDetail", "defaultKeyStatistics", "financialData"]
STATEMENT_MODULES = [
    "incomeStatementHistory", "balanceSheetHistory", "cashflowStatementHistory",
    "incomeStatementHistoryQuarterly", "balanceSheetHistoryQuarterly",
    "cashflowStatementHistoryQuarterly",
]
EARNINGS_MODULES = ["earningsHistory", "earningsTrend", "calendarEvents", "earnings"]
OWNERSHIP_MODULES = [
    "institutionOwnership", "fundOwnership", "majorHoldersBreakdown",
    "insiderHolders", "insiderTransactions", "upgradeDowngradeHistory",
]
RECOMMENDATION_MODULES = ["recommendationTrend"]
ESG_MODULES = ["esgScores"]
ALL_MODULES = [... all of the above ...]
```

### API

```python
async def quote_summary_async(
    symbols: str | list[str],
    *,
    modules: list[str] | str = "assetProfile",
    client: YahooClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch quoteSummary modules and return as deterministic Arrow table.

    For profile-type modules (assetProfile, quoteType, etc.), returns
    one row per symbol with static fields as columns.

    For statement-type modules, returns one row per (symbol, asOfDate)
    with financial fields as columns.

    For history-type modules (upgradeDowngradeHistory, insiderTransactions),
    returns one row per dated event.
    """


def quote_summary(
    symbols: str | list[str],
    *,
    modules: list[str] | str = "assetProfile",
    client: YahooClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper."""
```

### Convenience wrappers

```python
async def asset_profile_async(symbols, ...) -> pa.Table
async def summary_detail_async(symbols, ...) -> pa.Table
async def key_statistics_async(symbols, ...) -> pa.Table
async def financial_data_async(symbols, ...) -> pa.Table
async def calendar_events_async(symbols, ...) -> pa.Table
async def earnings_history_async(symbols, ...) -> pa.Table
async def upgrade_downgrade_history_async(symbols, ...) -> pa.Table
async def recommendation_trend_async(symbols, ...) -> pa.Table
async def institution_ownership_async(symbols, ...) -> pa.Table
async def insider_transactions_async(symbols, ...) -> pa.Table
```

### Implementation notes

- 1 symbol per request
- Response is deeply nested JSON — flatten to wide Arrow table
- Different modules produce different schemas — use module-specific parsers
- Statement modules return arrays of dated dicts → normalize to rows
- `upgradeDowngradeHistory` returns up to 14 years → most valuable historical module

---

## Implementation order

### Phase 1: fundamentals.py (highest value)
1. Write `src/yfin/fundamentals.py` with types constants + `fundamentals_async` + sync wrapper
2. Write `src/yfin/arrow.py` extension: `build_fundamentals_table(raw, types)`
3. Write tests: `tests/test_fundamentals.py`
4. End-to-end test against live Yahoo for AAPL

### Phase 2: summary.py (profile + stats)
1. Write `src/yfin/summary.py` with module constants + `quote_summary_async` + sync wrapper
2. Write per-module parsers in arrow.py: `build_profile_table`, `build_stats_table`
3. Write tests: `tests/test_summary.py`
4. End-to-end test

### Phase 3: summary.py (statements + earnings + ownership)
1. Extend `summary.py` with statement/earnings/ownership module parsers
2. Write tests for each module type
3. End-to-end test

### Phase 4: Convenience wrappers + `__init__.py` exports
1. Add convenience async/sync wrappers
2. Export from `yfin.__init__`
3. Full test suite green
4. Commit + push to GitHub main

---

## Quality gates
- `uv run pytest -q` — all tests pass
- `uv run ruff format --check . && uv run ruff check .`
- Live end-to-end test for AAPL, MSFT, NVDA
- No new dependencies (uses existing pyarrow, fastreq)
