"""Tests for the fundamentals-timeseries provider.

Uses Yahoo fundamentals-timeseries JSON fixtures covering: single symbol/type,
multi-symbol concatenation, multi-type columns, empty results, missing values,
integer-valued types, failed-symbol skipping, and the sync wrapper.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from yfin.arrow import build_fundamentals_table
from yfin.fundamentals import (
    _parse_timeseries,
    fundamentals,
    fundamentals_async,
)
from yfin.models import YahooRoute

# ---------------------------------------------------------------------------
# Yahoo fundamentals-timeseries fixtures
# ---------------------------------------------------------------------------

# One symbol, one type (PeRatio), 2 timestamps.
TS_SINGLE_TYPE = {
    "timeseries": {
        "result": [
            {
                "meta": {"symbol": "AAPL", "type": ["PeRatio"]},
                "timestamp": [1718841600, 1720742400],
                "PeRatio": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 30.03}},
                    {"asOfDate": "2024-07-12", "reportedValue": {"raw": 35.87}},
                ],
            }
        ]
    }
}

# One symbol, three types, 2 timestamps.
TS_MULTI_TYPE = {
    "timeseries": {
        "result": [
            {
                "meta": {"symbol": "AAPL", "type": ["PeRatio", "PbRatio", "PsRatio"]},
                "timestamp": [1718841600, 1720742400],
                "PeRatio": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 30.03}},
                    {"asOfDate": "2024-07-12", "reportedValue": {"raw": 35.87}},
                ],
                "PbRatio": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 39.91}},
                    {"asOfDate": "2024-07-12", "reportedValue": {"raw": 41.12}},
                ],
                "PsRatio": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 8.14}},
                    {"asOfDate": "2024-07-12", "reportedValue": {"raw": 8.77}},
                ],
            }
        ]
    }
}

# One symbol, integer-valued type (MarketCap), 2 timestamps.
TS_INTEGER_TYPE = {
    "timeseries": {
        "result": [
            {
                "meta": {"symbol": "AAPL", "type": ["MarketCap"]},
                "timestamp": [1718841600, 1720742400],
                "MarketCap": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 2950000000000}},
                    {"asOfDate": "2024-07-12", "reportedValue": {"raw": 3010000000000}},
                ],
            }
        ]
    }
}

# One symbol where one type is missing a value at one date.
TS_MISSING_VALUE = {
    "timeseries": {
        "result": [
            {
                "meta": {"symbol": "AAPL", "type": ["PeRatio", "PbRatio"]},
                "timestamp": [1718841600, 1720742400],
                "PeRatio": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 30.03}},
                    {"asOfDate": "2024-07-12", "reportedValue": {"raw": 35.87}},
                ],
                "PbRatio": [
                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 39.91}},
                    # PbRatio entry at index 1 missing -> null
                ],
            }
        ]
    }
}

# Empty result list.
TS_EMPTY = {"timeseries": {"result": []}}


# ---------------------------------------------------------------------------
# build_fundamentals_table (unit tests on the Arrow builder)
# ---------------------------------------------------------------------------


class TestBuildFundamentalsTable:
    def test_single_symbol_single_type(self) -> None:
        rows = _parse_timeseries(TS_SINGLE_TYPE, "AAPL")
        table = build_fundamentals_table(rows, types=["PeRatio"])
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.column_names == ["symbol", "as_of_date", "pe_ratio"]

    def test_values_correct(self) -> None:
        rows = _parse_timeseries(TS_SINGLE_TYPE, "AAPL")
        table = build_fundamentals_table(rows, types=["PeRatio"])
        pe = table.column("pe_ratio").to_pylist()
        assert pe == [30.03, 35.87]

    def test_dates_are_date32(self) -> None:
        rows = _parse_timeseries(TS_SINGLE_TYPE, "AAPL")
        table = build_fundamentals_table(rows, types=["PeRatio"])
        date_type = table.schema.field("as_of_date").type
        assert date_type == pa.date32()
        dates = table.column("as_of_date").to_pylist()
        assert dates == [dt.date(2024, 6, 20), dt.date(2024, 7, 12)]

    def test_multi_type_three_columns(self) -> None:
        rows = _parse_timeseries(TS_MULTI_TYPE, "AAPL")
        table = build_fundamentals_table(rows, types=["PeRatio", "PbRatio", "PsRatio"])
        assert table.column_names == [
            "symbol",
            "as_of_date",
            "pe_ratio",
            "pb_ratio",
            "ps_ratio",
        ]
        assert table.num_rows == 2

    def test_empty_result_has_schema(self) -> None:
        rows = _parse_timeseries(TS_EMPTY, "AAPL")
        table = build_fundamentals_table(rows, types=["PeRatio"])
        assert table.num_rows == 0
        assert table.column_names == ["symbol", "as_of_date", "pe_ratio"]
        assert table.schema.field("symbol").type == pa.string()
        assert table.schema.field("as_of_date").type == pa.date32()
        assert table.schema.field("pe_ratio").type == pa.float64()

    def test_missing_values_become_nulls(self) -> None:
        rows = _parse_timeseries(TS_MISSING_VALUE, "AAPL")
        table = build_fundamentals_table(rows, types=["PeRatio", "PbRatio"])
        pb = table.column("pb_ratio").to_pylist()
        assert pb[0] == 39.91
        assert pb[1] is None

    def test_integer_types_are_int64(self) -> None:
        rows = _parse_timeseries(TS_INTEGER_TYPE, "AAPL")
        table = build_fundamentals_table(rows, types=["MarketCap"])
        assert table.schema.field("market_cap").type == pa.int64()
        mc = table.column("market_cap").to_pylist()
        assert mc == [2950000000000, 3010000000000]


# ---------------------------------------------------------------------------
# _parse_timeseries
# ---------------------------------------------------------------------------


class TestParseTimeseries:
    def test_returns_rows_with_symbol(self) -> None:
        rows = _parse_timeseries(TS_SINGLE_TYPE, "AAPL")
        assert len(rows) == 2
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["asOfDate"] == "2024-06-20"
        assert rows[0]["PeRatio"] == 30.03

    def test_uses_meta_symbol_when_present(self) -> None:
        rows = _parse_timeseries(TS_SINGLE_TYPE, "REQUESTED")
        # meta.symbol = "AAPL" should win
        assert rows[0]["symbol"] == "AAPL"

    def test_empty_result(self) -> None:
        rows = _parse_timeseries(TS_EMPTY, "AAPL")
        assert rows == []

    def test_non_dict_response(self) -> None:
        assert _parse_timeseries(None, "AAPL") == []
        assert _parse_timeseries([], "AAPL") == []
        assert _parse_timeseries("nope", "AAPL") == []


# ---------------------------------------------------------------------------
# fundamentals_async with mock client
# ---------------------------------------------------------------------------


class _MockFundamentalsClient:
    """Mock client that returns different responses per symbol."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        # Extract symbol from URL tail.
        symbol = url.rsplit("/", 1)[-1].upper()
        if symbol not in self._responses:
            raise RuntimeError(f"unexpected symbol {symbol}")
        return self._responses[symbol]

    async def close(self) -> None:
        pass


class _ErrorClient:
    """Mock that raises on get_json for a specific symbol."""

    def __init__(self, responses: dict[str, dict], error_symbol: str, error: Exception) -> None:
        self._responses = responses
        self._error_symbol = error_symbol
        self._error = error

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        symbol = url.rsplit("/", 1)[-1].upper()
        if symbol == self._error_symbol:
            raise self._error
        return self._responses[symbol]

    async def close(self) -> None:
        pass


class TestFundamentalsAsync:
    async def test_single_symbol_single_type(self) -> None:
        client = _MockFundamentalsClient({"AAPL": TS_SINGLE_TYPE})
        table = await fundamentals_async(["AAPL"], types=["PeRatio"], client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.column_names == ["symbol", "as_of_date", "pe_ratio"]

    async def test_multi_symbol(self) -> None:
        client = _MockFundamentalsClient(
            {
                "AAPL": TS_SINGLE_TYPE,
                "MSFT": {
                    "timeseries": {
                        "result": [
                            {
                                "meta": {"symbol": "MSFT", "type": ["PeRatio"]},
                                "timestamp": [1718841600],
                                "PeRatio": [
                                    {"asOfDate": "2024-06-20", "reportedValue": {"raw": 36.5}},
                                ],
                            }
                        ]
                    }
                },
            }
        )
        table = await fundamentals_async(["AAPL", "MSFT"], types=["PeRatio"], client=client)
        assert table.num_rows == 3  # 2 AAPL + 1 MSFT
        symbols = table.column("symbol").to_pylist()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    async def test_multi_type(self) -> None:
        client = _MockFundamentalsClient({"AAPL": TS_MULTI_TYPE})
        table = await fundamentals_async(
            ["AAPL"], types=["PeRatio", "PbRatio", "PsRatio"], client=client
        )
        assert table.column_names == [
            "symbol",
            "as_of_date",
            "pe_ratio",
            "pb_ratio",
            "ps_ratio",
        ]
        assert table.num_rows == 2

    async def test_empty_result(self) -> None:
        client = _MockFundamentalsClient({"AAPL": TS_EMPTY})
        table = await fundamentals_async(["AAPL"], types=["PeRatio"], client=client)
        assert table.num_rows == 0
        # Schema still present.
        assert table.column_names == ["symbol", "as_of_date", "pe_ratio"]

    async def test_missing_values(self) -> None:
        client = _MockFundamentalsClient({"AAPL": TS_MISSING_VALUE})
        table = await fundamentals_async(["AAPL"], types=["PeRatio", "PbRatio"], client=client)
        pb = table.column("pb_ratio").to_pylist()
        assert pb[0] == 39.91
        assert pb[1] is None

    async def test_integer_types(self) -> None:
        client = _MockFundamentalsClient({"AAPL": TS_INTEGER_TYPE})
        table = await fundamentals_async(["AAPL"], types=["MarketCap"], client=client)
        assert table.schema.field("market_cap").type == pa.int64()
        mc = table.column("market_cap").to_pylist()
        assert mc == [2950000000000, 3010000000000]

    async def test_skips_failed_symbol(self) -> None:
        client = _ErrorClient(
            {"AAPL": TS_SINGLE_TYPE},
            error_symbol="MSFT",
            error=RuntimeError("simulated network failure"),
        )
        table = await fundamentals_async(["AAPL", "MSFT"], types=["PeRatio"], client=client)
        # AAPL rows present, MSFT failed and was skipped.
        assert table.num_rows == 2
        symbols = table.column("symbol").to_pylist()
        assert all(s == "AAPL" for s in symbols)

    async def test_accepts_single_symbol_string(self) -> None:
        client = _MockFundamentalsClient({"AAPL": TS_SINGLE_TYPE})
        table = await fundamentals_async("AAPL", types=["PeRatio"], client=client)
        assert table.num_rows == 2


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class TestFundamentalsSync:
    def test_sync_returns_same_as_async(self) -> None:
        # The async path returns 2 rows for TS_SINGLE_TYPE; sync must match.
        client = _MockFundamentalsClient({"AAPL": TS_SINGLE_TYPE})
        table = fundamentals(["AAPL"], types=["PeRatio"], client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.column_names == ["symbol", "as_of_date", "pe_ratio"]
        pe = table.column("pe_ratio").to_pylist()
        assert pe == [30.03, 35.87]

    def test_sync_fails_inside_running_loop(self) -> None:
        import asyncio

        async def _run_in_loop() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                fundamentals(
                    ["AAPL"],
                    types=["PeRatio"],
                    client=_MockFundamentalsClient({"AAPL": TS_SINGLE_TYPE}),
                )

        asyncio.run(_run_in_loop())
