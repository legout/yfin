"""Fundamentals timeseries provider.

Uses Yahoo ``query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/
timeseries/{symbol}`` (one request per symbol). Returns deterministic
``pyarrow.Table``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pyarrow as pa

from .arrow import build_fundamentals_table
from .client import YahooClient
from .models import YahooRoute, normalize_symbols
from .quotes import QuoteClient

__all__ = [
    "fundamentals_async",
    "fundamentals",
    "VALUATION_TYPES",
    "INCOME_STATEMENT_TYPES",
    "BALANCE_SHEET_TYPES",
    "CASH_FLOW_TYPES",
]

# ---------------------------------------------------------------------------
# Constants — Yahoo fundamentals-timeseries type groups
# ---------------------------------------------------------------------------

VALUATION_TYPES: list[str] = [
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

INCOME_STATEMENT_TYPES: list[str] = [
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

BALANCE_SHEET_TYPES: list[str] = [
    "TotalAssets",
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

CASH_FLOW_TYPES: list[str] = [
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
]

# All known types, for convenience.
ALL_TYPES: list[str] = (
    VALUATION_TYPES + INCOME_STATEMENT_TYPES + BALANCE_SHEET_TYPES + CASH_FLOW_TYPES
)

_FUNDAMENTALS_URL = (
    "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"
)

# Yahoo ignores period1 earlier than ~4 years; default to that window.
_DEFAULT_LOOKBACK_YEARS = 4


# ---------------------------------------------------------------------------
# Parameter builder
# ---------------------------------------------------------------------------


def build_fundamentals_params(
    symbol: str,
    types: list[str],
    *,
    start: dt.date | dt.datetime | int | None = None,
    end: dt.date | dt.datetime | int | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the fundamentals URL and query params for *symbol*.

    Returns ``(url, params)``.
    """
    if not types:
        raise ValueError("At least one type is required")

    period1, period2 = _resolve_period(start, end)

    params: dict[str, str] = {
        "period1": str(period1),
        "period2": str(period2),
        "type": ",".join(types),
        "merge": "false",
        "padTimeSeries": "false",
    }
    url = _FUNDAMENTALS_URL.format(symbol=symbol)
    return url, params


def _resolve_period(
    start: dt.date | dt.datetime | int | None,
    end: dt.date | dt.datetime | int | None,
) -> tuple[int, int]:
    """Resolve (period1, period2) epoch seconds with fundamentals defaults."""
    now = dt.datetime.now(dt.UTC)
    if start is not None:
        period1 = _to_epoch(start)
    else:
        period1 = int(now.replace(year=now.year - _DEFAULT_LOOKBACK_YEARS).timestamp())
    period2 = _to_epoch(end) if end is not None else int(now.timestamp())
    if period1 > period2:
        raise ValueError(f"start ({period1}) must not be after end ({period2})")
    return period1, period2


def _to_epoch(value: dt.date | dt.datetime | int) -> int:
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"Timestamp must be non-negative, got {value}")
        return value
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return int(value.timestamp())
    return int(dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC).timestamp())


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


async def fundamentals_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    types: list[str],
    start: dt.date | dt.datetime | int | None = None,
    end: dt.date | dt.datetime | int | None = None,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch fundamentals timeseries and return a deterministic Arrow table.

    Parameters
    ----------
    symbols
        One or more ticker symbols. Normalised, de-duplicated, order-preserved.
    types
        Yahoo fundamentals type names (camelCase). Use one of
        :data:`VALUATION_TYPES`, :data:`INCOME_STATEMENT_TYPES`,
        :data:`BALANCE_SHEET_TYPES`, :data:`CASH_FLOW_TYPES`, or
        :data:`ALL_TYPES`, or build a custom list.
    start / end
        Date range as ``datetime.date``, ``datetime.datetime``, or epoch
        seconds. Defaults to the last 4 years → now.
    client
        Reuse an existing :class:`YahooClient`.
    proxy
        Optional proxy URL for this request's route.
    """
    normalised = normalize_symbols(symbols)
    own_client = client is None
    if own_client:
        client = YahooClient(proxies=[proxy] if proxy else None)

    try:
        tasks = [
            _fetch_one(
                client,
                symbol,
                client.get_route(proxy),
                types=types,
                start=start,
                end=end,
            )
            for symbol in normalised
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for sym, result in zip(normalised, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{sym}: {type(result).__name__}: {result}")
            else:
                rows.extend(result)

        if errors:
            import logging

            logging.getLogger("yfin").warning(
                "Failed fundamentals symbols (%d/%d): %s",
                len(errors),
                len(normalised),
                ", ".join(e.split(":")[0] for e in errors),
            )

        return build_fundamentals_table(rows, types=types)
    finally:
        if own_client:
            await client.close()


async def _fetch_one(
    client: QuoteClient,
    symbol: str,
    route: YahooRoute,
    *,
    types: list[str],
    start: dt.date | dt.datetime | int | None,
    end: dt.date | dt.datetime | int | None,
) -> list[dict[str, Any]]:
    """Fetch and parse fundamentals for a single symbol into flat rows."""
    url, params = build_fundamentals_params(symbol, types, start=start, end=end)
    resp = await client.get_json(url, params=params, route=route)
    return _parse_timeseries(resp, symbol)


def _parse_timeseries(resp: Any, symbol: str) -> list[dict[str, Any]]:
    """Parse a Yahoo fundamentals-timeseries response into flat rows.

    Each row is ``{symbol, asOfDate, <Type1>: value, <Type2>: value, ...}``.
    The shared ``timestamp`` array indexes all type arrays positionally; a row
    is emitted for each timestamp, carrying whichever types have a value at
    that index.
    """
    if not isinstance(resp, dict):
        return []

    timeseries = resp.get("timeseries")
    if not isinstance(timeseries, dict):
        return []

    results = timeseries.get("result")
    if not isinstance(results, list):
        return []

    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        meta = result.get("meta")
        resolved_symbol = symbol
        if isinstance(meta, dict) and isinstance(meta.get("symbol"), str):
            resolved_symbol = meta["symbol"]

        # Identify which types are present in this result object.
        meta_types = meta.get("type") if isinstance(meta, dict) else None
        present_types: list[str] = []
        if isinstance(meta_types, list):
            for t in meta_types:
                if isinstance(t, str) and t in result:
                    present_types.append(t)
        else:
            # Fall back to scanning result keys that are lists of dicts.
            for key, val in result.items():
                if key in ("meta", "timestamp"):
                    continue
                if isinstance(val, list):
                    present_types.append(key)

        timestamps: list[int] = result.get("timestamp") or []
        type_arrays: dict[str, list[dict[str, Any] | None]] = {}
        for t in present_types:
            arr = result.get(t)
            type_arrays[t] = arr if isinstance(arr, list) else []

        n = len(timestamps)
        for i in range(n):
            row: dict[str, Any] = {"symbol": resolved_symbol}
            as_of: str | None = None
            for t in present_types:
                entry = _safe_index(type_arrays[t], i)
                if not isinstance(entry, dict):
                    continue
                if as_of is None:
                    as_of = entry.get("asOfDate")
                raw = _extract_reported_value(entry)
                if raw is not None:
                    row[t] = raw
            if as_of is not None:
                row["asOfDate"] = as_of
                rows.append(row)

    return rows


def _safe_index(lst: list[Any], i: int) -> Any:
    if 0 <= i < len(lst):
        return lst[i]
    return None


def _extract_reported_value(entry: dict[str, Any]) -> Any:
    """Pull ``reportedValue.raw`` (or top-level ``raw``) from a type entry."""
    rv = entry.get("reportedValue")
    if isinstance(rv, dict) and "raw" in rv:
        return rv["raw"]
    if "raw" in entry:
        return entry["raw"]
    return None


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def fundamentals(
    symbols: str | list[str] | tuple[str, ...],
    *,
    types: list[str],
    start: dt.date | dt.datetime | int | None = None,
    end: dt.date | dt.datetime | int | None = None,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Synchronous wrapper for :func:`fundamentals_async`.

    Raises :class:`RuntimeError` when called inside a running event loop.
    """
    _assert_no_running_loop()
    return asyncio.run(
        fundamentals_async(
            symbols,
            types=types,
            start=start,
            end=end,
            client=client,
            proxy=proxy,
        )
    )


def _assert_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "yfin sync wrappers must not be called from a running event loop. "
        "Use the async variant (fundamentals_async) instead."
    )
