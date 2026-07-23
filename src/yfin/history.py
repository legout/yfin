"""Chart history provider.

Uses Yahoo ``query1.finance.yahoo.com/v8/finance/chart/{symbol}`` (one request
per symbol). Returns deterministic ``pyarrow.Table``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from .arrow import build_history_table
from .client import YahooClient
from .models import YahooRoute, normalize_symbols, validate_date_range
from .quotes import QuoteClient

__all__ = ["history_async", "history"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_DEFAULT_EVENTS: tuple[str, ...] = ("div", "split")


# ---------------------------------------------------------------------------
# Parameter builder
# ---------------------------------------------------------------------------


def build_chart_params(
    symbol: str,
    *,
    start: dt.date | dt.datetime | int | None = None,
    end: dt.date | dt.datetime | int | None = None,
    period: str | None = None,
    interval: str = "1d",
    events: Sequence[str] = _DEFAULT_EVENTS,
    include_pre_post: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build the chart URL and query params for *symbol*.

    Returns ``(url, params)``. Raises :class:`YahooValidationError` on bad input.
    """
    date_range = validate_date_range(start, end, period, interval)

    params: dict[str, Any] = {
        "interval": interval,
        "events": ",".join(events) if events else "",
        "includePrePost": "true" if include_pre_post else "false",
    }

    if date_range is not None:
        p1, p2 = date_range
        params["period1"] = str(p1)
        params["period2"] = str(p2)
    elif period is not None:
        params["range"] = period
    else:
        # Default: full available history
        params["period1"] = "0"
        params["period2"] = str(int(dt.datetime.now(dt.UTC).timestamp()))

    url = _CHART_URL.format(symbol=symbol)
    return url, params


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


async def history_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    start: dt.date | dt.datetime | int | None = None,
    end: dt.date | dt.datetime | int | None = None,
    period: str | None = None,
    interval: str = "1d",
    events: Sequence[str] = _DEFAULT_EVENTS,
    include_pre_post: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Fetch historical OHLCV data and return a deterministic Arrow table.

    Parameters
    ----------
    symbols
        One or more ticker symbols. Normalised, de-duplicated, order-preserved.
    start / end
        Date range as ``datetime.date``, ``datetime.datetime``, or epoch seconds.
        Cannot be combined with ``period``.
    period
        Yahoo range string (e.g. ``"1y"``, ``"max"``). Cannot be combined
        with explicit dates.
    interval
        Bar interval (default ``"1d"``).
    events
        Event types to include (default ``("div", "split")``).
    include_pre_post
        Include pre/post market data (default ``False``).
    client
        Reuse an existing :class:`YahooClient`.
    proxy
        Optional proxy URL for this request's route.
    """
    normalised = normalize_symbols(symbols)
    own_client = client is None
    if own_client:
        client = YahooClient(proxies=[proxy] if proxy else None)

    route = client.get_route(proxy)

    try:
        tasks = [
            _fetch_one(
                client,
                symbol,
                route,
                start=start,
                end=end,
                period=period,
                interval=interval,
                events=events,
                include_pre_post=include_pre_post,
            )
            for symbol in normalised
        ]
        tables = await asyncio.gather(*tasks)

        if not tables:
            from .models import HISTORY_SCHEMA

            arrays = [pa.array([], type=field.type) for field in HISTORY_SCHEMA]
            return pa.table(arrays, schema=HISTORY_SCHEMA)

        return pa.concat_tables(tables)
    finally:
        if own_client:
            await client.close()


async def _fetch_one(
    client: QuoteClient,
    symbol: str,
    route: Any,
    **kwargs: Any,
) -> pa.Table:
    """Fetch and parse chart data for a single symbol."""
    url, params = build_chart_params(symbol, **kwargs)
    resp = await client.get_json(url, params=params, route=route)
    chart_result = _extract_chart_result(resp, symbol)
    return build_history_table(symbol, chart_result)


def _extract_chart_result(resp: Any, symbol: str) -> dict[str, Any] | None:
    """Extract ``chart.result[0]`` from a Yahoo v8 response."""
    if not isinstance(resp, dict):
        return None
    chart = resp.get("chart")
    if not isinstance(chart, dict):
        return None

    # Check for chart-level error
    error = chart.get("error")
    if isinstance(error, dict):
        from .exceptions import YahooApiError

        code = error.get("code", "Unknown")
        description = error.get("description", "")
        msg = f"{code}: {description}" if description else str(code)
        raise YahooApiError(f"Yahoo chart error for {symbol}: {msg}")

    result = chart.get("result")
    if not isinstance(result, list) or not result:
        return None
    first = result[0]
    if not isinstance(first, dict):
        return None
    return first


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def history(
    symbols: str | list[str] | tuple[str, ...],
    *,
    start: dt.date | dt.datetime | int | None = None,
    end: dt.date | dt.datetime | int | None = None,
    period: str | None = None,
    interval: str = "1d",
    events: Sequence[str] = _DEFAULT_EVENTS,
    include_pre_post: bool = False,
    client: QuoteClient | None = None,
    proxy: str | None = None,
) -> pa.Table:
    """Synchronous wrapper for :func:`history_async`.

    Raises :class:`RuntimeError` when called inside a running event loop.
    """
    from .quotes import _assert_no_running_loop

    _assert_no_running_loop()
    return asyncio.run(
        history_async(
            symbols,
            start=start,
            end=end,
            period=period,
            interval=interval,
            events=events,
            include_pre_post=include_pre_post,
            client=client,
            proxy=proxy,
        )
    )
