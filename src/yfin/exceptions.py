"""Typed exceptions for yfin.

All yfin exceptions inherit from :class:`YahooError` so callers can catch the
entire family with a single ``except`` clause.
"""

from __future__ import annotations

__all__ = [
    "YahooError",
    "YahooAuthError",
    "YahooCrumbError",
    "YahooConsentError",
    "YahooRateLimitError",
    "YahooApiError",
    "YahooSymbolError",
    "YahooValidationError",
]


class YahooError(Exception):
    """Base class for all yfin errors."""


class YahooAuthError(YahooError):
    """Cookie/crumb authentication failed and no strategy succeeded."""


class YahooCrumbError(YahooAuthError):
    """Yahoo returned a blank, HTML, or otherwise invalid crumb."""


class YahooConsentError(YahooAuthError):
    """CSRF consent flow failed (missing form fields, parse error, etc.)."""


class YahooRateLimitError(YahooError):
    """Yahoo responded with HTTP 429 / Too Many Requests."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class YahooApiError(YahooError):
    """Yahoo returned a structured error payload (``finance.error`` code)."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code
        super().__init__(message)


class YahooSymbolError(YahooError):
    """A symbol failed validation or normalisation."""


class YahooValidationError(YahooError):
    """Caller-supplied parameters are invalid (e.g. both period and date range)."""
