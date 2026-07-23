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

from .arrow import build_history_table, build_quote_table, to_polars
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

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # Public functions
    "quotes",
    "quotes_async",
    "history",
    "history_async",
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
