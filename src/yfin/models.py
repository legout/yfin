"""Core data models for yfin.

This module defines the deterministic Arrow schemas, symbol normalisation,
validation utilities, and small immutable value objects used throughout the
package.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import Enum

import pyarrow as pa

from .exceptions import YahooSymbolError, YahooValidationError

__all__ = [
    "Interval",
    "QuoteFields",
    "Range",
    "HISTORY_SCHEMA",
    "normalize_symbols",
    "camel_to_snake",
    "crumb_to_int_timestamps",
    "validate_date_range",
    "YahooRoute",
]

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

HISTORY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("timestamp", pa.timestamp("s", tz="UTC")),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("adjusted_close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("dividend", pa.float64()),
        pa.field("split_ratio", pa.float64()),
        pa.field("currency", pa.string()),
        pa.field("exchange_timezone", pa.string()),
    ]
)


class Interval(str, Enum):
    """Valid Yahoo chart intervals."""

    M1 = "1m"
    M2 = "2m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    M60 = "60m"
    M90 = "90m"
    H1 = "1h"
    D1 = "1d"
    D5 = "5d"
    WK1 = "1wk"
    MO1 = "1mo"
    MO3 = "3mo"


class Range(str, Enum):
    """Valid Yahoo chart ranges."""

    D1 = "1d"
    D5 = "5d"
    MO1 = "1mo"
    MO3 = "3mo"
    MO6 = "6mo"
    Y1 = "1y"
    Y2 = "2y"
    Y5 = "5y"
    Y10 = "10y"
    YTD = "ytd"
    MAX = "max"


class QuoteFields(str, Enum):
    """Common Yahoo v7 quote field names (camelCase as Yahoo expects)."""

    REGULAR_MARKET_PRICE = "regularMarketPrice"
    REGULAR_MARKET_VOLUME = "regularMarketVolume"
    REGULAR_MARKET_PREVIOUS_CLOSE = "regularMarketPreviousClose"
    REGULAR_MARKET_CHANGE = "regularMarketChange"
    REGULAR_MARKET_CHANGE_PERCENT = "regularMarketChangePercent"
    REGULAR_MARKET_DAY_HIGH = "regularMarketDayHigh"
    REGULAR_MARKET_DAY_LOW = "regularMarketDayLow"
    REGULAR_MARKET_OPEN = "regularMarketOpen"
    REGULAR_MARKET_TIME = "regularMarketTime"
    MARKET_CAP = "marketCap"
    CURRENCY = "currency"
    SHORT_NAME = "shortName"
    LONG_NAME = "longName"
    EXCHANGE = "exchange"
    EXCHANGE_NAME = "fullExchangeName"
    QUOTE_TYPE = "quoteType"
    SYMBOL = "symbol"
    FIFTY_TWO_WEEK_HIGH = "fiftyTwoWeekHigh"
    FIFTY_TWO_WEEK_LOW = "fiftyTwoWeekLow"
    TRAILING_PE = "trailingPE"
    FORWARD_PE = "forwardPE"
    EPS_TRAILING_TWELVE_MONTHS = "epsTrailingTwelveMonths"
    EPS_FORWARD = "epsForward"
    DIVIDEND_RATE = "dividendRate"
    DIVIDEND_YIELD = "dividendYield"
    TRAILING_ANNUAL_DIVIDEND_RATE = "trailingAnnualDividendRate"
    TRAILING_ANNUAL_DIVIDEND_YIELD = "trailingAnnualDividendYield"
    BETA = "beta"
    SHARES_OUTSTANDING = "sharesOutstanding"
    BOOK_VALUE = "bookValue"
    PRICE_TO_BOOK = "priceToBook"
    AVERAGE_DAILY_VOLUME_3MONTH = "averageDailyVolume3Month"
    AVERAGE_DAILY_VOLUME_10DAY = "averageDailyVolume10Day"


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------

# Yahoo symbols are 1-12 chars, uppercase, alphanumerics + .-^=
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^=]+$")


def normalize_symbols(symbols: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalise and validate a symbol or list of symbols.

    - Accept a single ``str`` or any sequence of strings.
    - Strip whitespace and uppercase each symbol.
    - De-duplicate while preserving the first-occurrence order.
    - Raise :class:`YahooSymbolError` on empty input or invalid characters.
    """
    if isinstance(symbols, str):
        raw: list[str] = [symbols]
    else:
        raw = list(symbols)

    if not raw:
        raise YahooSymbolError("At least one symbol is required")

    normalised: list[str] = []
    seen: set[str] = set()
    for s in raw:
        if not isinstance(s, str):
            raise YahooSymbolError(f"Symbol must be a string, got {type(s).__name__}: {s!r}")
        cleaned = s.strip().upper()
        if not cleaned:
            raise YahooSymbolError("Symbols must not be empty after stripping whitespace")
        if not _SYMBOL_RE.match(cleaned):
            raise YahooSymbolError(
                f"Invalid symbol {cleaned!r}: only A-Z, 0-9, '.', '-', '^', '=' are allowed"
            )
        if len(cleaned) > 12:
            raise YahooSymbolError(f"Symbol {cleaned!r} exceeds 12 characters")
        if cleaned not in seen:
            seen.add(cleaned)
            normalised.append(cleaned)

    return normalised


# ---------------------------------------------------------------------------
# Camel/snake conversion
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def camel_to_snake(name: str) -> str:
    """Convert a single camelCase identifier to snake_case."""
    s1 = _CAMEL_BOUNDARY_1.sub(r"\1_\2", name)
    return _CAMEL_BOUNDARY_2.sub(r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_INTERVALS = frozenset(i.value for i in Interval)
_VALID_RANGES = frozenset(r.value for r in Range)


def validate_date_range(
    start: dt.date | dt.datetime | int | None,
    end: dt.date | dt.datetime | int | None,
    period: str | None,
    interval: str,
) -> tuple[int, int] | None:
    """Validate caller parameters and return ``(period1, period2)`` epoch seconds.

    Returns ``None`` when a ``period`` (range) string should be used instead of
    explicit timestamps.

    Raises :class:`YahooValidationError` on conflicting or invalid input.
    """
    if interval not in _VALID_INTERVALS:
        raise YahooValidationError(
            f"Invalid interval {interval!r}; valid: {sorted(_VALID_INTERVALS)}"
        )

    using_dates = start is not None or end is not None

    if period is not None and using_dates:
        raise YahooValidationError("Cannot specify both explicit dates and period/range")

    if period is not None:
        if period not in _VALID_RANGES:
            raise YahooValidationError(
                f"Invalid period/range {period!r}; valid: {sorted(_VALID_RANGES)}"
            )
        return None

    if start is None and end is None:
        return None  # caller will default to range

    if start is not None and end is not None:
        p1 = _to_epoch(start)
        p2 = _to_epoch(end)
    elif start is not None:
        p1 = _to_epoch(start)
        p2 = int(dt.datetime.now(dt.UTC).timestamp())
    else:
        p1 = 0
        p2 = _to_epoch(end)  # type: ignore[arg-type]

    if p1 > p2:
        raise YahooValidationError(f"start ({p1}) must not be after end ({p2})")

    return p1, p2


def _to_epoch(value: dt.date | dt.datetime | int) -> int:
    if isinstance(value, int):
        if value < 0:
            raise YahooValidationError(f"Timestamp must be non-negative, got {value}")
        return value
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return int(value.timestamp())
    # date (not datetime)
    return int(dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC).timestamp())


# ---------------------------------------------------------------------------
# Yahoo error detection
# ---------------------------------------------------------------------------

_YAHOO_ERROR_CODES = frozenset({"Crumb validator failed", "Invalid Crumb"})


def detect_yahoo_error(payload: object) -> str | None:
    """Return a human-readable error description if *payload* is a Yahoo error.

    Yahoo error payloads have the shape::

        {"finance": {"error": {"code": "...", "description": "..."}}}

    Returns ``None`` when the payload does not look like a Yahoo error.
    """
    if not isinstance(payload, dict):
        return None
    finance = payload.get("finance")
    if not isinstance(finance, dict):
        return None
    err = finance.get("error")
    if not isinstance(err, dict):
        return None
    code = err.get("code", "Unknown")
    description = err.get("description", "")
    if isinstance(code, str) and isinstance(description, str):
        return f"{code}: {description}" if description else code
    return None


# ---------------------------------------------------------------------------
# Route / proxy identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YahooRoute:
    """Identifies a distinct network route to Yahoo.

    Two requests share a route iff they use the same proxy URL (``""`` for
    direct). Cookie/crumb state is cached per route so a crumb obtained through
    one proxy is never sent through another.
    """

    proxy: str = ""

    def __str__(self) -> str:
        return f"direct" if not self.proxy else f"proxy:{self.proxy}"


def crumb_to_int_timestamps(period1: int, period2: int) -> tuple[int, int]:
    """Pass-through helper kept for API symmetry with timestamp normalisation."""
    return period1, period2
