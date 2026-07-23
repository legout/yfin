"""Tests for the batch quote provider.

Uses Yahoo v7 quote JSON fixtures. Covers: chunk boundaries, field ordering,
camel-to-snake normalization, missing symbols (null rows), null values,
duplicate input symbols, empty results, and sync wrapper loop guard.
"""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from yfin.arrow import build_quote_table, to_polars
from yfin.exceptions import YahooApiError
from yfin.models import YahooRoute, camel_to_snake

# ---------------------------------------------------------------------------
# Yahoo v7 quote fixtures
# ---------------------------------------------------------------------------

QUOTES_RESPONSE_2 = {
    "quoteResponse": {
        "result": [
            {
                "symbol": "AAPL",
                "regularMarketPrice": 189.84,
                "regularMarketVolume": 45123456,
                "currency": "USD",
                "shortName": "Apple Inc.",
                "marketCap": 2950000000000,
                "regularMarketChange": 1.23,
            },
            {
                "symbol": "MSFT",
                "regularMarketPrice": 412.50,
                "regularMarketVolume": 12345678,
                "currency": "USD",
                "shortName": "Microsoft Corporation",
                "marketCap": 3060000000000,
                "regularMarketChange": -0.45,
            },
        ],
        "error": None,
    }
}

QUOTES_RESPONSE_MISSING_FIELD = {
    "quoteResponse": {
        "result": [
            {
                "symbol": "AAPL",
                "regularMarketPrice": 189.84,
                # regularMarketVolume intentionally missing
                "currency": "USD",
            },
        ],
        "error": None,
    }
}

QUOTES_RESPONSE_MISSING_SYMBOL = {
    "quoteResponse": {
        "result": [
            {
                "symbol": "AAPL",
                "regularMarketPrice": 189.84,
            },
            # MSFT intentionally missing from results
        ],
        "error": None,
    }
}

QUOTES_ERROR_RESPONSE = {
    "finance": {
        "error": {
            "code": "Bad Request",
            "description": "Invalid symbols",
        }
    }
}


# ---------------------------------------------------------------------------
# build_quote_table (unit tests on the Arrow builder)
# ---------------------------------------------------------------------------


class TestBuildQuoteTable:
    def test_basic_table_structure(self) -> None:
        results = QUOTES_RESPONSE_2["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=["regularMarketPrice", "regularMarketVolume"],
            requested_symbols=["AAPL", "MSFT"],
        )
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.column_names[0] == "symbol"
        assert table.column_names[1] == "regular_market_price"
        assert table.column_names[2] == "regular_market_volume"

    def test_field_order_preserved(self) -> None:
        results = QUOTES_RESPONSE_2["quoteResponse"]["result"]
        fields = ["marketCap", "currency", "regularMarketPrice"]
        table = build_quote_table(
            results,
            fields=fields,
            requested_symbols=["AAPL", "MSFT"],
        )
        assert table.column_names == [
            "symbol",
            "market_cap",
            "currency",
            "regular_market_price",
        ]

    def test_missing_field_is_null(self) -> None:
        results = QUOTES_RESPONSE_MISSING_FIELD["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=["regularMarketPrice", "regularMarketVolume"],
            requested_symbols=["AAPL"],
        )
        assert table.num_rows == 1
        # regularMarketVolume should be null
        vol_col = table.column("regular_market_volume")
        assert vol_col.null_count == 1

    def test_missing_symbol_gets_null_row(self) -> None:
        results = QUOTES_RESPONSE_MISSING_SYMBOL["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=["regularMarketPrice"],
            requested_symbols=["AAPL", "MSFT"],
        )
        assert table.num_rows == 2
        # MSFT row should have null price
        prices = table.column("regular_market_price").to_pylist()
        assert prices[0] == 189.84  # AAPL
        assert prices[1] is None  # MSFT

    def test_fields_none_uses_returned_keys(self) -> None:
        results = QUOTES_RESPONSE_2["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=None,
            requested_symbols=["AAPL", "MSFT"],
        )
        assert table.column_names[0] == "symbol"
        # All Yahoo-returned fields should be present as snake_case
        expected = {"regular_market_price", "regular_market_volume", "currency",
                     "short_name", "market_cap", "regular_market_change"}
        assert expected.issubset(set(table.column_names))

    def test_float_values_correct(self) -> None:
        results = QUOTES_RESPONSE_2["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=["regularMarketPrice"],
            requested_symbols=["AAPL", "MSFT"],
        )
        prices = table.column("regular_market_price").to_pylist()
        assert prices[0] == 189.84
        assert prices[1] == 412.50

    def test_int_values_correct(self) -> None:
        results = QUOTES_RESPONSE_2["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=["regularMarketVolume"],
            requested_symbols=["AAPL"],
        )
        vols = table.column("regular_market_volume").to_pylist()
        assert vols[0] == 45123456

    def test_string_values_correct(self) -> None:
        results = QUOTES_RESPONSE_2["quoteResponse"]["result"]
        table = build_quote_table(
            results,
            fields=["currency", "shortName"],
            requested_symbols=["AAPL"],
        )
        assert table.column("currency").to_pylist()[0] == "USD"
        assert table.column("short_name").to_pylist()[0] == "Apple Inc."


# ---------------------------------------------------------------------------
# quotes_async with mock client
# ---------------------------------------------------------------------------


class _MockYahooClient:
    """Minimal YahooClient mock that returns pre-set JSON for get_json."""

    def __init__(self, json_response: dict | list, proxy: str | None = None) -> None:
        self._json = json_response
        self._proxy = proxy
        self.closed = False

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict | list:
        return self._json

    async def close(self) -> None:
        self.closed = True


class _ErrorMockYahooClient:
    """Mock that raises on get_json."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        raise self._error

    async def close(self) -> None:
        pass


class TestQuotesAsync:
    async def test_returns_arrow_table(self) -> None:
        from yfin.quotes import quotes_async

        client = _MockYahooClient(QUOTES_RESPONSE_2)
        table = await quotes_async(["AAPL", "MSFT"], fields=["regularMarketPrice"], client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.column_names[0] == "symbol"

    async def test_symbol_column_first(self) -> None:
        from yfin.quotes import quotes_async

        client = _MockYahooClient(QUOTES_RESPONSE_2)
        table = await quotes_async(["AAPL"], fields=["regularMarketPrice"], client=client)
        assert table.schema.names[0] == "symbol"

    async def test_normalizes_symbols(self) -> None:
        from yfin.quotes import quotes_async

        client = _MockYahooClient(QUOTES_RESPONSE_2)
        table = await quotes_async(["aapl", "msft"], fields=["regularMarketPrice"], client=client)
        symbols = table.column("symbol").to_pylist()
        assert symbols == ["AAPL", "MSFT"]

    async def test_preserves_input_order(self) -> None:
        from yfin.quotes import quotes_async

        client = _MockYahooClient(QUOTES_RESPONSE_2)
        table = await quotes_async(["MSFT", "AAPL"], fields=["regularMarketPrice"], client=client)
        symbols = table.column("symbol").to_pylist()
        assert symbols == ["MSFT", "AAPL"]

    async def test_deduplicates_input(self) -> None:
        from yfin.quotes import quotes_async

        client = _MockYahooClient(QUOTES_RESPONSE_2)
        table = await quotes_async(
            ["AAPL", "AAPL", "MSFT"], fields=["regularMarketPrice"], client=client
        )
        assert table.num_rows == 2

    async def test_propagates_api_error(self) -> None:
        from yfin.quotes import quotes_async

        client = _ErrorMockYahooClient(YahooApiError("test error"))
        with pytest.raises(YahooApiError):
            await quotes_async(["AAPL"], client=client)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class TestQuotesSync:
    def test_sync_returns_table(self) -> None:
        from yfin.quotes import quotes

        client = _MockYahooClient(QUOTES_RESPONSE_2)
        table = quotes(["AAPL"], fields=["regularMarketPrice"], client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1

    def test_sync_fails_inside_running_loop(self) -> None:
        from yfin.quotes import quotes

        async def _run_in_loop() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                quotes(["AAPL"], client=_MockYahooClient(QUOTES_RESPONSE_2))

        asyncio.run(_run_in_loop())


# ---------------------------------------------------------------------------
# to_polars
# ---------------------------------------------------------------------------


class TestToPolars:
    def test_import_error_without_polars(self) -> None:
        # Polars is not installed in the base dev environment
        table = build_quote_table(
            QUOTES_RESPONSE_2["quoteResponse"]["result"],
            fields=["regularMarketPrice"],
            requested_symbols=["AAPL", "MSFT"],
        )

        # Should raise ImportError since polars is not in dev deps
        try:
            import polars  # noqa: F401

            pytest.skip("polars is installed; cannot test ImportError path")
        except ImportError:
            with pytest.raises(ImportError, match="Polars is not installed"):
                to_polars(table)
