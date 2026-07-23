"""Shared test fakes: a controllable fake fastreq response and request function.

These allow fully hermetic (offline) testing of auth, quotes, and history
without any network access.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from yfin.models import YahooRoute

__all__ = ["FakeResponse", "FakeRequestHandler", "make_request_func"]


@dataclass
class FakeResponse:
    """Mimics fastreq.NormalizedResponse for test purposes."""

    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""
    text: str = ""
    json_data: Any = None
    url: str = ""
    is_json: bool = False
    cookies: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text and not self.content:
            self.content = self.text.encode("utf-8")
        if self.json_data is not None:
            self.is_json = True


class FakeRequestHandler:
    """Programmable fake request handler for testing.

    Register URL patterns to responses, or queue responses in order.
    Captures all requests for assertion.
    """

    def __init__(self) -> None:
        self._responses: deque[FakeResponse] = deque()
        self._url_map: dict[str, FakeResponse] = {}
        self._call_log: list[dict[str, Any]] = []
        self._url_call_counts: dict[str, int] = {}

    def queue_response(self, response: FakeResponse) -> None:
        """Queue a response to be returned in FIFO order."""
        self._responses.append(response)

    def map_url(self, url_pattern: str, response: FakeResponse) -> None:
        """Map a URL (exact match) to a response."""
        self._url_map[url_pattern] = response

    def map_url_prefix(self, prefix: str, response: FakeResponse) -> None:
        """Map a URL prefix to a response."""
        self._url_map[prefix] = response

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return self._call_log

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    def url_call_count(self, url: str) -> int:
        return self._url_call_counts.get(url, 0)

    def find_call(self, url_substring: str) -> dict[str, Any] | None:
        for call in self._call_log:
            if url_substring in call.get("url", ""):
                return call
        return None

    async def handle(self, url: str, **kwargs: Any) -> FakeResponse:
        """Handle a request and return the registered/queued response."""
        self._call_log.append({"url": url, **kwargs})
        self._url_call_counts[url] = self._url_call_counts.get(url, 0) + 1

        # Try exact URL match
        if url in self._url_map:
            return self._url_map[url]

        # Try prefix match
        for pattern, resp in self._url_map.items():
            if url.startswith(pattern):
                return resp

        # Try queued response
        if self._responses:
            return self._responses.popleft()

        raise AssertionError(f"FakeRequestHandler has no response for URL: {url}")


def make_request_func(handler: FakeRequestHandler):
    """Create a request function suitable for YahooAuth."""

    async def _request_func(url: str, **kwargs: Any) -> FakeResponse:
        return await handler.handle(url, **kwargs)

    return _request_func


def make_json_response(
    json_data: Any,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    cookie: str | None = None,
) -> FakeResponse:
    """Convenience: build a FakeResponse with JSON data."""
    hdrs = headers or {}
    if cookie:
        hdrs["set-cookie"] = cookie
    return FakeResponse(
        status_code=status_code,
        json_data=json_data,
        is_json=True,
        headers=hdrs,
    )


def make_text_response(
    text: str,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    cookie: str | None = None,
) -> FakeResponse:
    """Convenience: build a FakeResponse with text body."""
    hdrs = headers or {}
    if cookie:
        hdrs["set-cookie"] = cookie
    return FakeResponse(
        status_code=status_code,
        text=text,
        headers=hdrs,
    )
