"""Batch quote provider.

Uses Yahoo ``query1.finance.yahoo.com/v7/finance/quote`` with bounded
comma-separated symbol chunks. Returns deterministic ``pyarrow.Table``.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa
from fastreq.utils.progress import ProgressCallback, ProgressOption, gather_with_progress

from .arrow import build_quote_table
from .client import YahooClient
from .models import YahooRoute, normalize_symbols

__all__ = ["quotes_async", "quotes", "QuoteClient"]

# ---------------------------------------------------------------------------
# Protocol for injectable clients (enables testing with fakes)
# ---------------------------------------------------------------------------


@runtime_checkable
class QuoteClient(Protocol):
    """Minimal interface a quote/history client must satisfy."""

    def get_route(self, proxy: str | None = None) -> YahooRoute: ...

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        route: YahooRoute | None = None,
    ) -> Any: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUOTE_URL = "https://query2.finance.yahoo.com/v7/finance/quote"
DEFAULT_CHUNK_SIZE = 200
MAX_URL_LENGTH = 8000  # conservative URL length guard


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_symbols(
    symbols: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[list[str]]:
    """Split *symbols* into URL-safe chunks respecting both count and URL size.

    Each chunk becomes a comma-separated ``symbols`` query parameter. We guard
    against the total URL exceeding :data:`MAX_URL_LENGTH`.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for sym in symbols:
        # +1 for comma separator (or the param key prefix for the first)
        sep = 1 if current else len("symbols=")
        encoded_len = len(urllib.parse.quote(sym, safe="")) + sep
        if current and (len(current) >= chunk_size or current_len + encoded_len > MAX_URL_LENGTH):
            chunks.append(current)
            current = []
            current_len = 0
        current.append(sym)
        current_len += encoded_len

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


async def quotes_async(
    symbols: str | list[str] | tuple[str, ...],
    *,
    fields: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    client: QuoteClient | None = None,
    proxy: str | None = None,
    progress: ProgressOption = None,
    progress_callback: ProgressCallback | None = None,
) -> pa.Table:
    """Fetch batch quotes and return a deterministic Arrow table.

    Parameters
    ----------
    symbols
        One or more ticker symbols. Normalised, de-duplicated, order-preserved.
    fields
        Yahoo quote fields in camelCase (e.g. ``["regularMarketPrice"]``).
        When ``None``, all fields Yahoo returns are included.
    chunk_size
        Maximum symbols per request (default 200).
    client
        Reuse an existing :class:`YahooClient` (e.g. for proxy pools). A new
        transient client is created when omitted.
    proxy
        Optional proxy URL for this request's route.
    progress
        Optional ``"rich"``/``"tqdm"`` progress bar, or ``True`` to auto-select.
    progress_callback
        Optional callback receiving ``(completed, total)``.
    """
    normalised = normalize_symbols(symbols)
    own_client = client is None
    if own_client:
        client = YahooClient(proxies=[proxy] if proxy else None)

    try:
        chunks = chunk_symbols(normalised, chunk_size)
        all_results: list[dict[str, Any]] = []

        tasks = [
            client.get_json(
                _QUOTE_URL,
                params={
                    "symbols": ",".join(chunk),
                    **_fields_params(fields),
                },
                route=client.get_route(proxy),
            )
            for chunk in chunks
        ]
        responses = await gather_with_progress(
            tasks,
            mode=progress,
            callback=progress_callback,
            description="Yahoo quote chunks",
            return_exceptions=False,
        )

        for resp in responses:
            results = _extract_quote_results(resp)
            all_results.extend(results)

        return build_quote_table(all_results, fields=fields, requested_symbols=normalised)
    finally:
        if own_client:
            await client.close()


def _fields_params(fields: Sequence[str] | None) -> dict[str, str]:
    if fields is None:
        return {}
    return {"fields": ",".join(fields)}


def _extract_quote_results(resp: Any) -> list[dict[str, Any]]:
    """Extract the result list from a Yahoo v7 quote response."""
    if not isinstance(resp, dict):
        return []
    quote_response = resp.get("quoteResponse")
    if not isinstance(quote_response, dict):
        return []
    result = quote_response.get("result")
    if not isinstance(result, list):
        return []
    return result


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def quotes(
    symbols: str | list[str] | tuple[str, ...],
    *,
    fields: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    client: QuoteClient | None = None,
    proxy: str | None = None,
    progress: ProgressOption = None,
    progress_callback: ProgressCallback | None = None,
) -> pa.Table:
    """Synchronous wrapper for :func:`quotes_async`.

    Raises :class:`RuntimeError` when called inside a running event loop.
    """
    _assert_no_running_loop()
    return asyncio.run(
        quotes_async(
            symbols,
            fields=fields,
            chunk_size=chunk_size,
            client=client,
            proxy=proxy,
            progress=progress,
            progress_callback=progress_callback,
        )
    )


def _assert_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return  # No running loop — safe to call asyncio.run()
    raise RuntimeError(
        "yfin sync wrappers must not be called from a running event loop. "
        "Use the async variant (quotes_async) instead."
    )
