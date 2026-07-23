"""Tests for symbol normalisation, chunking, and models.

Covers: camelCase->snake_case, symbol validation/dedup, date/period validation,
Yahoo error detection.
"""

from __future__ import annotations

import datetime as dt

import pytest

from yfin.exceptions import YahooSymbolError, YahooValidationError
from yfin.models import (
    YahooRoute,
    camel_to_snake,
    detect_yahoo_error,
    normalize_symbols,
    validate_date_range,
)
from yfin.quotes import chunk_symbols


# ---------------------------------------------------------------------------
# camel_to_snake
# ---------------------------------------------------------------------------


class TestCamelToSnake:
    def test_simple(self) -> None:
        assert camel_to_snake("regularMarketPrice") == "regular_market_price"

    def test_with_number(self) -> None:
        assert camel_to_snake("fiftyTwoWeekHigh") == "fifty_two_week_high"

    def test_already_snake(self) -> None:
        assert camel_to_snake("already_snake") == "already_snake"

    def test_single_word(self) -> None:
        assert camel_to_snake("price") == "price"

    def test_all_caps(self) -> None:
        assert camel_to_snake("URL") == "url"

    def test_consecutive_caps(self) -> None:
        assert camel_to_snake("HTTPResponse") == "http_response"

    def test_mixed_with_digits(self) -> None:
        assert camel_to_snake("epsTrailing12Months") == "eps_trailing12_months"


# ---------------------------------------------------------------------------
# normalize_symbols
# ---------------------------------------------------------------------------


class TestNormalizeSymbols:
    def test_single_string(self) -> None:
        assert normalize_symbols("aapl") == ["AAPL"]

    def test_list_of_strings(self) -> None:
        result = normalize_symbols(["aapl", "msft"])
        assert result == ["AAPL", "MSFT"]

    def test_strips_whitespace(self) -> None:
        assert normalize_symbols("  aapl  ") == ["AAPL"]

    def test_uppercase(self) -> None:
        assert normalize_symbols("aapl") == ["AAPL"]

    def test_dedup_preserves_order(self) -> None:
        result = normalize_symbols(["AAPL", "MSFT", "AAPL", "GOOG"])
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(YahooSymbolError, match="empty after stripping"):
            normalize_symbols("")

    def test_empty_list_raises(self) -> None:
        with pytest.raises(YahooSymbolError, match="At least one symbol"):
            normalize_symbols([])

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(YahooSymbolError, match="empty after stripping"):
            normalize_symbols("   ")

    def test_invalid_chars_raise(self) -> None:
        with pytest.raises(YahooSymbolError, match="Invalid symbol"):
            normalize_symbols("AAPL@")

    def test_space_in_symbol_raises(self) -> None:
        with pytest.raises(YahooSymbolError, match="Invalid symbol"):
            normalize_symbols("AAPL MSFT")

    def test_too_long_raises(self) -> None:
        with pytest.raises(YahooSymbolError, match="exceeds 12"):
            normalize_symbols("ABCDEFGHIJKLM")

    def test_caret_symbol_ok(self) -> None:
        assert normalize_symbols("^GSPC") == ["^GSPC"]

    def test_equals_symbol_ok(self) -> None:
        assert normalize_symbols("EUR=X") == ["EUR=X"]

    def test_dash_symbol_ok(self) -> None:
        assert normalize_symbols("BTC-USD") == ["BTC-USD"]

    def test_dot_symbol_ok(self) -> None:
        assert normalize_symbols("VOW.DE") == ["VOW.DE"]

    def test_non_string_raises(self) -> None:
        with pytest.raises(YahooSymbolError, match="must be a string"):
            normalize_symbols([123, "AAPL"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# validate_date_range
# ---------------------------------------------------------------------------


class TestValidateDateRange:
    def test_period_returns_none(self) -> None:
        assert validate_date_range(None, None, "1y", "1d") is None

    def test_period_and_dates_conflict(self) -> None:
        with pytest.raises(YahooValidationError, match="Cannot specify both"):
            validate_date_range(dt.date(2024, 1, 1), None, "1y", "1d")

    def test_invalid_interval(self) -> None:
        with pytest.raises(YahooValidationError, match="Invalid interval"):
            validate_date_range(None, None, None, "99d")

    def test_invalid_period(self) -> None:
        with pytest.raises(YahooValidationError, match="Invalid period"):
            validate_date_range(None, None, "99y", "1d")

    def test_date_objects(self) -> None:
        result = validate_date_range(dt.date(2024, 1, 1), dt.date(2024, 6, 1), None, "1d")
        assert result is not None
        p1, p2 = result
        assert p1 < p2
        assert p1 == 1704067200  # 2024-01-01 UTC

    def test_datetime_objects(self) -> None:
        result = validate_date_range(
            dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 6, 1, tzinfo=dt.UTC),
            None,
            "1d",
        )
        assert result is not None
        assert result[0] < result[1]

    def test_int_timestamps(self) -> None:
        result = validate_date_range(1704067200, 1717200000, None, "1d")
        assert result == (1704067200, 1717200000)

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(YahooValidationError, match="must not be after"):
            validate_date_range(dt.date(2024, 6, 1), dt.date(2024, 1, 1), None, "1d")

    def test_negative_timestamp_raises(self) -> None:
        with pytest.raises(YahooValidationError, match="non-negative"):
            validate_date_range(-1, 100, None, "1d")

    def test_start_only_defaults_end_to_now(self) -> None:
        result = validate_date_range(1704067200, None, None, "1d")
        assert result is not None
        assert result[0] == 1704067200
        assert result[1] > 1704067200

    def test_none_all_defaults_to_none(self) -> None:
        assert validate_date_range(None, None, None, "1d") is None


# ---------------------------------------------------------------------------
# detect_yahoo_error
# ---------------------------------------------------------------------------


class TestDetectYahooError:
    def test_no_error(self) -> None:
        assert detect_yahoo_error({"chart": {"result": []}}) is None

    def test_error_payload(self) -> None:
        payload = {
            "finance": {
                "error": {"code": "Bad Request", "description": "Invalid symbol"}
            }
        }
        result = detect_yahoo_error(payload)
        assert result is not None
        assert "Bad Request" in result
        assert "Invalid symbol" in result

    def test_error_no_description(self) -> None:
        payload = {"finance": {"error": {"code": "Error"}}}
        result = detect_yahoo_error(payload)
        assert result == "Error"

    def test_non_dict_returns_none(self) -> None:
        assert detect_yahoo_error("not a dict") is None
        assert detect_yahoo_error(None) is None
        assert detect_yahoo_error(42) is None

    def test_no_finance_key(self) -> None:
        assert detect_yahoo_error({"quoteResponse": {"result": []}}) is None


# ---------------------------------------------------------------------------
# YahooRoute
# ---------------------------------------------------------------------------


class TestYahooRoute:
    def test_direct_route(self) -> None:
        r = YahooRoute()
        assert r.proxy == ""
        assert str(r) == "direct"

    def test_proxy_route(self) -> None:
        r = YahooRoute(proxy="http://proxy:8080")
        assert r.proxy == "http://proxy:8080"
        assert str(r) == "proxy:http://proxy:8080"

    def test_equality(self) -> None:
        assert YahooRoute() == YahooRoute()
        assert YahooRoute(proxy="http://a") == YahooRoute(proxy="http://a")
        assert YahooRoute(proxy="http://a") != YahooRoute(proxy="http://b")

    def test_hashable(self) -> None:
        s = {YahooRoute(), YahooRoute(proxy="http://a")}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# chunk_symbols
# ---------------------------------------------------------------------------


class TestChunkSymbols:
    def test_single_chunk(self) -> None:
        chunks = chunk_symbols(["AAPL", "MSFT", "GOOG"])
        assert len(chunks) == 1
        assert chunks[0] == ["AAPL", "MSFT", "GOOG"]

    def test_respects_chunk_size(self) -> None:
        symbols = [f"S{i}" for i in range(5)]
        chunks = chunk_symbols(symbols, chunk_size=2)
        assert len(chunks) == 3
        assert chunks[0] == ["S0", "S1"]
        assert chunks[1] == ["S2", "S3"]
        assert chunks[2] == ["S4"]

    def test_empty_list(self) -> None:
        assert chunk_symbols([]) == []

    def test_single_symbol(self) -> None:
        assert chunk_symbols(["AAPL"]) == [["AAPL"]]

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_symbols(["AAPL"], chunk_size=0)

    def test_preserves_order(self) -> None:
        symbols = ["C", "A", "B", "D", "E"]
        chunks = chunk_symbols(symbols, chunk_size=2)
        flat = [s for chunk in chunks for s in chunk]
        assert flat == symbols

    def test_default_chunk_size(self) -> None:
        symbols = [f"S{i}" for i in range(201)]
        chunks = chunk_symbols(symbols)
        assert len(chunks) == 2
        assert len(chunks[0]) == 200
        assert len(chunks[1]) == 1
