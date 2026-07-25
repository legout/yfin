"""yfin — Compact Python 3.14 Yahoo Finance client returning pyarrow.Table.

Built on fastreq. Provides batch quotes and chart history as deterministic
``pyarrow.Table``, with optional Polars conversion via the ``polars`` extra.

Public API::

    import yfin

    quotes = yfin.quotes(["AAPL", "MSFT"], fields=["regularMarketPrice"])
    history = yfin.history(["AAPL"], period="1y")

    # Async equivalents:
    quotes = await yfin.quotes_async(["AAPL"])
    history = await yfin.history_async(["AAPL"], period="1y")

    # Optional Polars conversion:
    df = yfin.to_polars(history)  # requires: pip install 'yfin[polars]'
"""

from __future__ import annotations

from typing import Any

from .arrow import (
    build_fundamentals_table,
    build_history_table,
    build_quote_table,
    build_summary_table,
    to_polars,
)
from .auth import AuthStrategy, YahooAuth, YahooSessionState
from .client import YahooClient
from .exceptions import (
    YahooApiError,
    YahooAuthError,
    YahooConsentError,
    YahooCrumbError,
    YahooError,
    YahooRateLimitError,
    YahooSymbolError,
    YahooValidationError,
)
from .fundamentals import (
    ALL_TYPES,
    BALANCE_SHEET_TYPES,
    CASH_FLOW_TYPES,
    INCOME_STATEMENT_TYPES,
    VALUATION_TYPES,
    fundamentals,
    fundamentals_async,
)
from .history import history, history_async
from .models import (
    HISTORY_SCHEMA,
    Interval,
    QuoteFields,
    Range,
    YahooRoute,
    camel_to_snake,
    normalize_symbols,
)
from .quotes import quotes, quotes_async
from .summary import (
    ALL_SUMMARY_MODULES,
    ANALYST_MODULES,
    CALENDAR_MODULES,
    OWNERSHIP_MODULES,
    PROFILE_MODULES,
    STATS_MODULES,
    asset_profile,
    asset_profile_async,
    calendar_events,
    calendar_events_async,
    financial_data,
    financial_data_async,
    insider_transactions,
    insider_transactions_async,
    institution_ownership,
    institution_ownership_async,
    key_statistics,
    key_statistics_async,
    quote_summary,
    quote_summary_async,
    recommendation_trend,
    recommendation_trend_async,
    summary_detail,
    summary_detail_async,
    upgrade_downgrade_history,
    upgrade_downgrade_history_async,
)

__version__ = "1.1.0"

__all__ = [
    "__version__",
    # Public functions — history
    "history",
    "history_async",
    # Public functions — quotes
    "quotes",
    "quotes_async",
    # Public functions — fundamentals
    "fundamentals",
    "fundamentals_async",
    # Public functions — summary
    "quote_summary",
    "quote_summary_async",
    "asset_profile",
    "asset_profile_async",
    "summary_detail",
    "summary_detail_async",
    "key_statistics",
    "key_statistics_async",
    "financial_data",
    "financial_data_async",
    "calendar_events",
    "calendar_events_async",
    "upgrade_downgrade_history",
    "upgrade_downgrade_history_async",
    "recommendation_trend",
    "recommendation_trend_async",
    "institution_ownership",
    "institution_ownership_async",
    "insider_transactions",
    "insider_transactions_async",
    # Constants
    "VALUATION_TYPES",
    "INCOME_STATEMENT_TYPES",
    "BALANCE_SHEET_TYPES",
    "CASH_FLOW_TYPES",
    "ALL_TYPES",
    "PROFILE_MODULES",
    "STATS_MODULES",
    "CALENDAR_MODULES",
    "ANALYST_MODULES",
    "OWNERSHIP_MODULES",
    "ALL_SUMMARY_MODULES",
    # Polars
    "to_polars",
    # Client
    "YahooClient",
    # Auth
    "YahooAuth",
    "YahooSessionState",
    "AuthStrategy",
    # Models / schemas
    "HISTORY_SCHEMA",
    "Interval",
    "QuoteFields",
    "Range",
    "YahooRoute",
    "normalize_symbols",
    "camel_to_snake",
    # Arrow builders (advanced)
    "build_quote_table",
    "build_history_table",
    "build_fundamentals_table",
    "build_summary_table",
    # Exceptions
    "YahooError",
    "YahooAuthError",
    "YahooCrumbError",
    "YahooConsentError",
    "YahooRateLimitError",
    "YahooApiError",
    "YahooSymbolError",
    "YahooValidationError",
]

# Type alias for documentation purposes
Table = Any  # pyarrow.Table — kept as Any to avoid importing pyarrow at package level
