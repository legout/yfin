"""quoteSummary v10 provider.

Uses Yahoo ``query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}``
(one request per symbol, modules batched in a single comma-separated query
parameter). Returns deterministic ``pyarrow.Table``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pyarrow as pa

from .arrow import build_summary_table
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
    "ALL_SUMMARY_MODULES",
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
    "earningHistory",
]
OWNERSHIP_MODULES = [
    "institutionOwnership",
    "fundOwnership",
    "majorHoldersBreakdown",
    "insiderHolders",
    "insiderTransactions",
]
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
        for convenience (e.g. ``"assetProfile"``).
    client
        Reuse an existing :class:`YahooClient` (e.g. for proxy pools). A new
        transient client is created when omitted.
    proxy
        Optional proxy URL for this request's route.
    """
    module_list = [modules] if isinstance(modules, str) else list(modules)
    if not module_list:
        raise ValueError("At least one module is required")

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

        return build_summary_table(raw_data, module_list)
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
    return _extract_summary_result(resp, symbol)


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
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch assetProfile + quoteType modules."""
    return await quote_summary_async(symbols, PROFILE_MODULES, client=client, proxy=proxy)


async def summary_detail_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch summaryDetail + defaultKeyStatistics + financialData modules."""
    return await quote_summary_async(symbols, STATS_MODULES, client=client, proxy=proxy)


async def key_statistics_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the defaultKeyStatistics module."""
    return await quote_summary_async(symbols, ["defaultKeyStatistics"], client=client, proxy=proxy)


async def financial_data_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the financialData module."""
    return await quote_summary_async(symbols, ["financialData"], client=client, proxy=proxy)


async def calendar_events_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the calendarEvents module."""
    return await quote_summary_async(symbols, CALENDAR_MODULES, client=client, proxy=proxy)


async def upgrade_downgrade_history_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the upgradeDowngradeHistory module."""
    return await quote_summary_async(
        symbols, ["upgradeDowngradeHistory"], client=client, proxy=proxy
    )


async def recommendation_trend_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the recommendationTrend module."""
    return await quote_summary_async(symbols, ["recommendationTrend"], client=client, proxy=proxy)


async def institution_ownership_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the institutionOwnership module."""
    return await quote_summary_async(symbols, ["institutionOwnership"], client=client, proxy=proxy)


async def insider_transactions_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch the insiderTransactions module."""
    return await quote_summary_async(symbols, ["insiderTransactions"], client=client, proxy=proxy)


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


def quote_summary(
    symbols: str | list[str] | tuple[str, ...],
    modules: list[str] | str,
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Synchronous wrapper for :func:`quote_summary_async`."""
    _assert_no_running_loop()
    return asyncio.run(quote_summary_async(symbols, modules, client=client, proxy=proxy))


def asset_profile(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`asset_profile_async`."""
    _assert_no_running_loop()
    return asyncio.run(asset_profile_async(symbols, client=client, proxy=proxy))


def summary_detail(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`summary_detail_async`."""
    _assert_no_running_loop()
    return asyncio.run(summary_detail_async(symbols, client=client, proxy=proxy))


def key_statistics(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`key_statistics_async`."""
    _assert_no_running_loop()
    return asyncio.run(key_statistics_async(symbols, client=client, proxy=proxy))


def financial_data(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`financial_data_async`."""
    _assert_no_running_loop()
    return asyncio.run(financial_data_async(symbols, client=client, proxy=proxy))


def calendar_events(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`calendar_events_async`."""
    _assert_no_running_loop()
    return asyncio.run(calendar_events_async(symbols, client=client, proxy=proxy))


def upgrade_downgrade_history(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`upgrade_downgrade_history_async`."""
    _assert_no_running_loop()
    return asyncio.run(upgrade_downgrade_history_async(symbols, client=client, proxy=proxy))


def recommendation_trend(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`recommendation_trend_async`."""
    _assert_no_running_loop()
    return asyncio.run(recommendation_trend_async(symbols, client=client, proxy=proxy))


def institution_ownership(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`institution_ownership_async`."""
    _assert_no_running_loop()
    return asyncio.run(institution_ownership_async(symbols, client=client, proxy=proxy))


def insider_transactions(
    symbols: str | list[str] | tuple[str, ...],
    *,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Sync wrapper for :func:`insider_transactions_async`."""
    _assert_no_running_loop()
    return asyncio.run(insider_transactions_async(symbols, client=client, proxy=proxy))


def _assert_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "yfin sync wrappers must not be called from a running event loop. "
        "Use the async variant (quote_summary_async) instead."
    )
