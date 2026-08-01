"""Tests for the chart history provider.

Uses Yahoo v8 chart JSON fixtures covering: normal bars, missing volumes,
null OHLC values, adjusted close, dividends, splits, Yahoo error payloads,
and a symbol with no data.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from yfin.arrow import build_history_table
from yfin.exceptions import YahooApiError, YahooValidationError
from yfin.history import build_chart_params, history_async
from yfin.models import HISTORY_SCHEMA, YahooRoute

# ---------------------------------------------------------------------------
# Yahoo v8 chart fixtures
# ---------------------------------------------------------------------------

CHART_NORMAL = {
    "chart": {
        "result": [
            {
                "meta": {
                    "currency": "USD",
                    "exchangeTimezoneName": "America/New_York",
                    "symbol": "AAPL",
                    "instrumentType": "EQUITY",
                },
                "timestamp": [1609459200, 1609545600, 1609632000],
                "indicators": {
                    "quote": [
                        {
                            "open": [129.41, 132.05, 131.25],
                            "high": [132.99, 133.40, 135.49],
                            "low": [129.12, 131.04, 130.93],
                            "close": [132.69, 131.97, 134.18],
                            "volume": [134883400, 99091400, 98891400],
                        }
                    ],
                    "adjclose": [{"adjclose": [132.02, 131.30, 133.50]}],
                },
                "events": {
                    "dividends": {
                        "1609545600": {
                            "date": 1609545600,
                            "amount": 0.205,
                        }
                    },
                    "splits": {},
                },
            }
        ],
        "error": None,
    }
}

CHART_WITH_SPLIT = {
    "chart": {
        "result": [
            {
                "meta": {
                    "currency": "USD",
                    "exchangeTimezoneName": "America/New_York",
                },
                "timestamp": [1609459200, 1609545600],
                "indicators": {
                    "quote": [
                        {
                            "open": [100.0, 50.0],
                            "high": [105.0, 52.0],
                            "low": [98.0, 49.0],
                            "close": [102.0, 51.0],
                            "volume": [1000000, 2000000],
                        }
                    ],
                },
                "events": {
                    "splits": {
                        "1609545600": {
                            "date": 1609545600,
                            "numerator": 2,
                            "denominator": 1,
                            "splitRatio": "2:1",
                        }
                    },
                    "dividends": {},
                },
            }
        ],
        "error": None,
    }
}

CHART_NULL_OHLC = {
    "chart": {
        "result": [
            {
                "meta": {"currency": "USD", "exchangeTimezoneName": "America/New_York"},
                "timestamp": [1609459200, 1609545600],
                "indicators": {
                    "quote": [
                        {
                            "open": [129.41, None],
                            "high": [132.99, None],
                            "low": [129.12, None],
                            "close": [132.69, None],
                            "volume": [134883400, None],
                        }
                    ],
                },
            }
        ],
        "error": None,
    }
}

CHART_NO_VOLUME = {
    "chart": {
        "result": [
            {
                "meta": {"currency": "EUR", "exchangeTimezoneName": "Europe/Berlin"},
                "timestamp": [1609459200],
                "indicators": {
                    "quote": [
                        {
                            "open": [10.0],
                            "high": [10.5],
                            "low": [9.8],
                            "close": [10.2],
                            "volume": [None],
                        }
                    ],
                },
            }
        ],
        "error": None,
    }
}

CHART_NO_DATA = {
    "chart": {
        "result": [None],
        "error": None,
    }
}

CHART_ERROR = {
    "chart": {
        "result": None,
        "error": {
            "code": "Not Found",
            "description": "No data found, symbol may be delisted",
        },
    }
}

CHART_NO_RESULT_KEY = {
    "chart": {
        "error": None,
    }
}


# ---------------------------------------------------------------------------
# build_history_table (Arrow builder unit tests)
# ---------------------------------------------------------------------------


class TestBuildHistoryTable:
    def test_normal_bars(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)

        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
        assert table.schema == HISTORY_SCHEMA

    def test_symbol_column(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        symbols = table.column("symbol").to_pylist()
        assert all(s == "AAPL" for s in symbols)

    def test_ohlc_values(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        opens = table.column("open").to_pylist()
        highs = table.column("high").to_pylist()
        lows = table.column("low").to_pylist()
        closes = table.column("close").to_pylist()

        assert opens == [129.41, 132.05, 131.25]
        assert highs == [132.99, 133.40, 135.49]
        assert lows == [129.12, 131.04, 130.93]
        assert closes == [132.69, 131.97, 134.18]

    def test_adjusted_close(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        adj = table.column("adjusted_close").to_pylist()
        assert adj == [132.02, 131.30, 133.50]

    def test_volume(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        vols = table.column("volume").to_pylist()
        assert vols == [134883400, 99091400, 98891400]

    def test_currency(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        currencies = table.column("currency").to_pylist()
        assert all(c == "USD" for c in currencies)

    def test_exchange_timezone(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        tzs = table.column("exchange_timezone").to_pylist()
        assert all(tz == "America/New_York" for tz in tzs)

    def test_dividend(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        divs = table.column("dividend").to_pylist()
        # Day 2 has a dividend
        assert divs[1] == 0.205
        # Days 1 and 3 don't
        assert divs[0] is None
        assert divs[2] is None

    def test_split_ratio_from_numerator_denominator(self) -> None:
        result = CHART_WITH_SPLIT["chart"]["result"][0]
        table = build_history_table("TEST", result)
        splits = table.column("split_ratio").to_pylist()
        assert splits[0] is None  # No split on day 1
        assert splits[1] == 2.0  # 2:1 split on day 2

    def test_null_ohlc_preserved(self) -> None:
        result = CHART_NULL_OHLC["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        opens = table.column("open").to_pylist()
        assert opens[0] == 129.41
        assert opens[1] is None

    def test_null_volume_is_null_not_zero(self) -> None:
        result = CHART_NO_VOLUME["chart"]["result"][0]
        table = build_history_table("EURS", result)
        vols = table.column("volume").to_pylist()
        assert vols[0] is None  # Not coerced to 0

    def test_no_data_returns_empty_table(self) -> None:
        table = build_history_table("NONE", None)
        assert table.num_rows == 0
        assert table.schema == HISTORY_SCHEMA

    def test_empty_timestamp_returns_empty_table(self) -> None:
        result = {"meta": {}, "timestamp": [], "indicators": {}}
        table = build_history_table("NONE", result)
        assert table.num_rows == 0

    def test_timestamps_are_utc(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        ts_type = table.schema.field("timestamp").type
        assert ts_type.tz == "UTC"

    def test_timestamps_values(self) -> None:
        result = CHART_NORMAL["chart"]["result"][0]
        table = build_history_table("AAPL", result)
        timestamps = table.column("timestamp").to_pylist()
        assert timestamps[0] == dt.datetime(2021, 1, 1, tzinfo=dt.UTC)
        assert timestamps[1] == dt.datetime(2021, 1, 2, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# build_chart_params
# ---------------------------------------------------------------------------


class TestBuildChartParams:
    def test_period_range(self) -> None:
        url, params = build_chart_params("AAPL", period="1y")
        assert url == "https://query2.finance.yahoo.com/v8/finance/chart/AAPL"
        assert params["range"] == "1y"
        assert params["interval"] == "1d"
        assert "period1" not in params
        assert "period2" not in params

    def test_date_range(self) -> None:
        url, params = build_chart_params(
            "AAPL",
            start=dt.date(2024, 1, 1),
            end=dt.date(2024, 6, 1),
        )
        assert "period1" in params
        assert "period2" in params
        assert params["period1"] == "1704067200"
        assert int(params["period2"]) > int(params["period1"])

    def test_events_default(self) -> None:
        url, params = build_chart_params("AAPL", period="1y")
        assert "div" in params["events"]
        assert "split" in params["events"]

    def test_include_pre_post(self) -> None:
        url, params = build_chart_params("AAPL", period="1y", include_pre_post=True)
        assert params["includePrePost"] == "true"

    def test_exclude_pre_post(self) -> None:
        url, params = build_chart_params("AAPL", period="1y", include_pre_post=False)
        assert params["includePrePost"] == "false"

    def test_custom_interval(self) -> None:
        url, params = build_chart_params("AAPL", period="1mo", interval="1h")
        assert params["interval"] == "1h"

    def test_custom_events(self) -> None:
        url, params = build_chart_params("AAPL", period="1y", events=("div",))
        assert params["events"] == "div"

    def test_period_and_dates_conflict(self) -> None:
        with pytest.raises(YahooValidationError, match="Cannot specify both"):
            build_chart_params("AAPL", start=dt.date(2024, 1, 1), period="1y")

    def test_invalid_interval(self) -> None:
        with pytest.raises(YahooValidationError, match="Invalid interval"):
            build_chart_params("AAPL", period="1y", interval="99x")


# ---------------------------------------------------------------------------
# history_async with mock client
# ---------------------------------------------------------------------------


class _MockHistoryClient:
    """Mock client that returns different responses per symbol."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses
        self.closed = False

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        # Extract symbol from URL
        symbol = url.rsplit("/", 1)[-1].upper()
        return self._responses.get(symbol, CHART_NO_DATA)

    async def close(self) -> None:
        self.closed = False


class _ErrorHistoryClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        raise self._error

    async def close(self) -> None:
        pass


class TestHistoryAsync:
    async def test_returns_arrow_table(self) -> None:
        client = _MockHistoryClient({"AAPL": CHART_NORMAL})
        table = await history_async(["AAPL"], period="1y", client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
        assert table.schema == HISTORY_SCHEMA

    async def test_multiple_symbols_concatenated(self) -> None:
        client = _MockHistoryClient(
            {
                "AAPL": CHART_NORMAL,
                "MSFT": {
                    "chart": {
                        "result": [
                            {
                                "meta": {
                                    "currency": "USD",
                                    "exchangeTimezoneName": "America/New_York",
                                },
                                "timestamp": [1609459200],
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": [200],
                                            "high": [210],
                                            "low": [195],
                                            "close": [205],
                                            "volume": [5000000],
                                        }
                                    ]
                                },
                            }
                        ],
                        "error": None,
                    }
                },
            }
        )
        table = await history_async(["AAPL", "MSFT"], period="1y", client=client)
        assert table.num_rows == 4  # 3 AAPL + 1 MSFT
        symbols = table.column("symbol").to_pylist()
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    async def test_preserves_input_order(self) -> None:
        client = _MockHistoryClient({"AAPL": CHART_NORMAL, "MSFT": CHART_NORMAL})
        table = await history_async(["MSFT", "AAPL"], period="1y", client=client)
        symbols = table.column("symbol").to_pylist()
        # MSFT should appear before AAPL in the concatenated output
        msft_first = symbols.index("MSFT")
        aapl_first = symbols.index("AAPL")
        assert msft_first < aapl_first

    async def test_reports_progress(self) -> None:
        completed: list[tuple[int, int]] = []
        client = _MockHistoryClient({"AAPL": CHART_NORMAL, "MSFT": CHART_NORMAL})
        await history_async(
            ["AAPL", "MSFT"],
            period="1y",
            client=client,
            progress_callback=lambda current, total: completed.append((current, total)),
        )
        assert completed == [(1, 2), (2, 2)]

    async def test_error_response_raises(self) -> None:
        client = _MockHistoryClient({"BAD": CHART_ERROR})
        with pytest.raises(YahooApiError, match="Not Found"):
            await history_async(["BAD"], period="1y", client=client)

    async def test_no_data_returns_empty_rows(self) -> None:
        client = _MockHistoryClient({"NONE": CHART_NO_DATA})
        table = await history_async(["NONE"], period="1y", client=client)
        assert table.num_rows == 0
        assert table.schema == HISTORY_SCHEMA

    async def test_propagates_api_error(self) -> None:
        client = _ErrorHistoryClient(YahooApiError("network failure"))
        with pytest.raises(YahooApiError):
            await history_async(["AAPL"], period="1y", client=client)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class TestHistorySync:
    def test_sync_returns_table(self) -> None:
        from yfin.history import history

        client = _MockHistoryClient({"AAPL": CHART_NORMAL})
        table = history(["AAPL"], period="1y", client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3

    def test_sync_fails_inside_running_loop(self) -> None:
        import asyncio

        from yfin.history import history

        async def _run_in_loop() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                history(["AAPL"], period="1y", client=_MockHistoryClient({"AAPL": CHART_NORMAL}))

        asyncio.run(_run_in_loop())
