"""quoteSummary v10 provider.

Uses Yahoo ``query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}``
(one request per symbol, modules batched in a single comma-separated query
parameter). Returns deterministic ``pyarrow.Table``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pyarrow as pa

from .arrow import (
    REQUESTED_SYMBOL_KEY,
    SUMMARY_MODULES,
    build_summary_table,
    strip_summary_prefixes,
)
from .client import YahooClient
from .exceptions import YahooApiError
from .models import YahooRoute, normalize_symbols
from .quotes import QuoteClient

__all__ = [
    # Module groups
    "PROFILE_MODULES",
    "STATS_MODULES",
    "CALENDAR_MODULES",
    "ANALYST_MODULES",
    "OWNERSHIP_MODULES",
    "INSIDER_TRANSACTIONS_MODULES",
    "ALL_SUMMARY_MODULES",
    # Module registry
    "SUMMARY_MODULES",
    # Generic entrypoint
    "quote_summary_async",
    "quote_summary",
    # Convenience wrappers (async)
    "asset_profile_async",
    "summary_detail_async",
    "key_statistics_async",
    "financial_data_async",
    "calendar_events_async",
    "upgrade_downgrade_history_async",
    "recommendation_trend_async",
    "institution_ownership_async",
    "insider_transactions_async",
    "fund_ownership_async",
    "insider_holders_async",
    "major_holders_breakdown_async",
    "major_direct_holders_async",
    "net_share_purchase_activity_async",
    "earnings_async",
    "earnings_history_async",
    # Convenience wrappers (sync)
    "asset_profile",
    "summary_detail",
    "key_statistics",
    "financial_data",
    "calendar_events",
    "upgrade_downgrade_history",
    "recommendation_trend",
    "institution_ownership",
    "insider_transactions",
    "fund_ownership",
    "insider_holders",
    "major_holders_breakdown",
    "major_direct_holders",
    "net_share_purchase_activity",
    "earnings",
    "earnings_history",
    # Column helpers
    "strip_summary_prefixes",
]

# ---------------------------------------------------------------------------
# Module groups
# ---------------------------------------------------------------------------

PROFILE_MODULES = ["assetProfile", "quoteType"]
STATS_MODULES = ["summaryDetail", "defaultKeyStatistics", "financialData"]
CALENDAR_MODULES = ["calendarEvents"]
ANALYST_MODULES = [
    "upgradeDowngradeHistory",
    "recommendationTrend",
    "earningsHistory",
]
OWNERSHIP_MODULES = [
    "institutionOwnership",
    "fundOwnership",
    "majorHoldersBreakdown",
    "insiderHolders",
    "insiderTransactions",
]
# Dedicated group for the insiderTransactions array module (parallel to
# CALENDAR_MODULES for the calendarEvents module).
INSIDER_TRANSACTIONS_MODULES = ["insiderTransactions"]
# Earnings & trends group: the earnings chart modules plus index/sector
# valuation trends and page-view momentum.
TREND_MODULES = ["indexTrend", "sectorTrend", "pageViews"]
ALL_SUMMARY_MODULES = [
    *PROFILE_MODULES,
    *STATS_MODULES,
    *CALENDAR_MODULES,
    *ANALYST_MODULES,
    *OWNERSHIP_MODULES,
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


async def quote_summary_async(
    symbols: str | list[str] | tuple[str, ...],
    modules: list[str] | str,
    *,
    strip_prefix: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch quoteSummary data and return a deterministic Arrow table.

    Parameters
    ----------
    symbols
        One or more ticker symbols. Normalised, de-duplicated, order-preserved.
    modules
        Yahoo quoteSummary module names. A single module string is accepted
        for convenience (e.g. ``"assetProfile"``). See
        :data:`yfin.SUMMARY_MODULES` for every module yfin can materialise.
    strip_prefix
        When True, drop the ``<module>_`` column-name prefix for the
        requested modules (``insider_transactions_shares`` → ``shares``).
        Raises :class:`ValueError` when stripping would merge distinct
        columns into one (possible when several modules share field names,
        e.g. ``maxAge``). Defaults to False (prefixed columns).
    client
        Reuse an existing :class:`YahooClient` (e.g. for proxy pools). A new
        transient client is created when omitted.
    proxy
        Optional proxy URL for this request's route.
    """
    module_list = [modules] if isinstance(modules, str) else list(modules)
    if not module_list:
        raise ValueError("At least one module is required")

    unknown = [m for m in module_list if m not in SUMMARY_MODULES]
    if unknown:
        raise ValueError(
            f"Unknown quoteSummary module(s): {unknown}. "
            f"Valid modules: {SUMMARY_MODULES}"
        )

    normalised = normalize_symbols(symbols)
    own_client = client is None
    if own_client:
        client = YahooClient(proxies=[proxy] if proxy else None)

    try:
        route = client.get_route(proxy)
        tasks = [_fetch_one(client, symbol, route, module_list) for symbol in normalised]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_data: list[dict[str, Any]] = []
        errors: list[str] = []
        for sym, result in zip(normalised, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{sym}: {type(result).__name__}: {result}")
            else:
                raw_data.append(result)

        if errors:
            import logging

            logging.getLogger("yfin").warning(
                "Failed quoteSummary symbols (%d/%d): %s",
                len(errors),
                len(normalised),
                ", ".join(e.split(":")[0] for e in errors),
            )

        if not raw_data:
            if errors:
                raise YahooApiError(
                    f"All {len(normalised)} symbols failed: {'; '.join(errors[:3])}"
                )
            return build_summary_table([], module_list)

        table = build_summary_table(raw_data, module_list)
        if strip_prefix:
            table = strip_summary_prefixes(table, module_list)
        return table
    finally:
        if own_client:
            await client.close()


async def _fetch_one(
    client: QuoteClient,
    symbol: str,
    route: YahooRoute,
    modules: list[str],
) -> dict[str, Any]:
    """Fetch and parse quoteSummary data for a single symbol.

    Returns the ``quoteSummary.result[0]`` dict (possibly empty when Yahoo
    returns no result).
    """
    url = _SUMMARY_URL.format(symbol=symbol)
    params = {"modules": ",".join(modules)}
    resp = await client.get_json(url, params=params, route=route)
    result = _extract_summary_result(resp, symbol)
    # Carry the requested symbol as authoritative provenance for the Arrow
    # builder. Narrow requests (insiderTransactions / calendarEvents only)
    # return no symbol-bearing module, so without this the flattener would
    # emit an empty ``symbol`` for every row.
    result[REQUESTED_SYMBOL_KEY] = symbol
    return result


def _extract_summary_result(resp: Any, symbol: str) -> dict[str, Any]:
    """Extract ``quoteSummary.result[0]`` from a Yahoo v10 response."""
    if not isinstance(resp, dict):
        return {}
    qs = resp.get("quoteSummary")
    if not isinstance(qs, dict):
        return {}

    error = qs.get("error")
    if isinstance(error, dict):
        code = error.get("code", "Unknown")
        description = error.get("description", "")
        msg = f"{code}: {description}" if description else str(code)
        raise YahooApiError(f"Yahoo quoteSummary error for {symbol}: {msg}")

    result = qs.get("result")
    if not isinstance(result, list) or not result:
        return {}
    first = result[0]
    if not isinstance(first, dict):
        return {}
    return first


# ---------------------------------------------------------------------------
# Convenience wrappers (async)
# ---------------------------------------------------------------------------


async def asset_profile_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch assetProfile + quoteType modules.

    ``strip_prefix`` defaults to False because the two modules share field
    names (``maxAge``) — stripping both would collide.
    """
    return await quote_summary_async(
        symbols, PROFILE_MODULES, strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def summary_detail_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch summaryDetail + defaultKeyStatistics + financialData modules.

    ``strip_prefix`` defaults to False because the three modules share field
    names (``maxAge``) — stripping all would collide.
    """
    return await quote_summary_async(
        symbols, STATS_MODULES, strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def key_statistics_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the defaultKeyStatistics module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["defaultKeyStatistics"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def financial_data_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the financialData module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["financialData"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def calendar_events_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the calendarEvents module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, CALENDAR_MODULES, strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def upgrade_downgrade_history_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the upgradeDowngradeHistory module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["upgradeDowngradeHistory"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def recommendation_trend_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the recommendationTrend module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["recommendationTrend"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def institution_ownership_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the institutionOwnership module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["institutionOwnership"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def insider_transactions_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """``insiderTransactions`` module (one row per filed transaction).

    Column prefix is stripped by default (``shares`` instead of
    ``insider_transactions_shares``).

    Note: Yahoo returns only the recent Form 4 history (typically a few
    dozen rows per symbol). Suitable for continuous capture, not for a
    comprehensive historical backfill — for that, use SEC EDGAR directly.
    """
    return await quote_summary_async(
        symbols,
        INSIDER_TRANSACTIONS_MODULES,
        strip_prefix=strip_prefix,
        client=client,
        proxy=proxy,
    )


async def fund_ownership_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the fundOwnership module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["fundOwnership"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def insider_holders_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the insiderHolders module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["insiderHolders"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def major_holders_breakdown_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the majorHoldersBreakdown module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["majorHoldersBreakdown"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def major_direct_holders_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the majorDirectHolders module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols, ["majorDirectHolders"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def net_share_purchase_activity_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the netSharePurchaseActivity module (prefix stripped by default)."""
    return await quote_summary_async(
        symbols,
        ["netSharePurchaseActivity"],
        strip_prefix=strip_prefix,
        client=client,
        proxy=proxy,
    )


async def earnings_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the ``earnings`` chart module (prefix stripped by default).

    Returns the earnings-chart estimate vs actual series plus the
    financials chart (revenue/expense history). Complex nested values are
    dropped by the flat path; the EPS trend survives as scalar fields.
    """
    return await quote_summary_async(
        symbols, ["earnings"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


async def earnings_history_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the earningsHistory module (prefix stripped by default).

    One row per reported quarter: EPS estimate vs actual, difference, and
    surprise percent.
    """
    return await quote_summary_async(
        symbols, ["earningsHistory"], strip_prefix=strip_prefix, client=client, proxy=proxy
    )


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


def quote_summary(
    symbols: str | list[str] | tuple[str, ...],
    modules: list[str] | str,
    *,
    strip_prefix: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Synchronous wrapper for :func:`quote_summary_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        quote_summary_async(
            symbols, modules, strip_prefix=strip_prefix, client=client, proxy=proxy
        )
    )


def asset_profile(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`asset_profile_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        asset_profile_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def summary_detail(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`summary_detail_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        summary_detail_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def key_statistics(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`key_statistics_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        key_statistics_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def financial_data(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`financial_data_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        financial_data_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def calendar_events(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`calendar_events_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        calendar_events_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def upgrade_downgrade_history(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`upgrade_downgrade_history_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        upgrade_downgrade_history_async(
            symbols, strip_prefix=strip_prefix, client=client, proxy=proxy
        )
    )


def recommendation_trend(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`recommendation_trend_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        recommendation_trend_async(
            symbols, strip_prefix=strip_prefix, client=client, proxy=proxy
        )
    )


def institution_ownership(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`institution_ownership_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        institution_ownership_async(
            symbols, strip_prefix=strip_prefix, client=client, proxy=proxy
        )
    )


def insider_transactions(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`insider_transactions_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        insider_transactions_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def fund_ownership(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`fund_ownership_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        fund_ownership_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def insider_holders(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`insider_holders_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        insider_holders_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def major_holders_breakdown(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`major_holders_breakdown_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        major_holders_breakdown_async(
            symbols, strip_prefix=strip_prefix, client=client, proxy=proxy
        )
    )


def major_direct_holders(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`major_direct_holders_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        major_direct_holders_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def net_share_purchase_activity(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`net_share_purchase_activity_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        net_share_purchase_activity_async(
            symbols, strip_prefix=strip_prefix, client=client, proxy=proxy
        )
    )


def earnings(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`earnings_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        earnings_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def earnings_history(
    symbols: str | list[str] | tuple[str, ...],
    *,
    strip_prefix: bool = True,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`earnings_history_async`."""
    _assert_no_running_loop()
    return asyncio.run(
        earnings_history_async(symbols, strip_prefix=strip_prefix, client=client, proxy=proxy)
    )


def _assert_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "yfin sync wrappers must not be called from a running event loop. "
        "Use the async variant (quote_summary_async) instead."
    )
