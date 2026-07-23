"""Integration tests for YahooClient and reliability scenarios.

Covers: 429/Retry-After handling, auth strategy switch through client,
structured errors, proxy route isolation at the client level, and client
cleanup. All tests are hermetic (no network).
"""

from __future__ import annotations

import pytest

from yfin.auth import YahooAuth
from yfin.client import YahooClient
from yfin.exceptions import YahooApiError, YahooRateLimitError
from yfin.models import YahooRoute

from .conftest import (
    FakeRequestHandler,
    FakeResponse,
    make_text_response,
)

FC_YAHOO = "https://fc.yahoo.com/"
GETCRUMB_Q1 = "https://query1.finance.yahoo.com/v1/test/getcrumb"
GETCRUMB_Q2 = "https://query2.finance.yahoo.com/v1/test/getcrumb"
GUCE_CONSENT = "https://guce.yahoo.com/consent"
CONSENT_COLLECT = "https://consent.yahoo.com/v2/collectConsent?sessionId=session-1"
COPYCONSENT = "https://guce.yahoo.com/copyConsent?sessionId=session-1"
QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
CHART_URL_PREFIX = "https://query1.finance.yahoo.com/v8/finance/chart/"


# ---------------------------------------------------------------------------
# Client-level test: build a YahooClient with a fake fastreq session
# ---------------------------------------------------------------------------


class _FakeFastReq:
    """Minimal fake of fastreq.FastRequests for YahooClient tests."""

    def __init__(self, handler: FakeRequestHandler) -> None:
        self._handler = handler

    async def request(self, url: str, **kwargs: object) -> FakeResponse:
        # YahooClient passes return_type='response'; we always return FakeResponse.
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "return_type"}
        return await self._handler.handle(url, **kwargs_clean)

    async def close(self) -> None:
        pass


def make_client_with_handler(handler: FakeRequestHandler) -> YahooClient:
    """Build a YahooClient whose fastreq session is a _FakeFastReq."""
    client = YahooClient.__new__(YahooClient)
    client._fastreq = _FakeFastReq(handler)
    client._proxies = []
    client._proxy_index = 0
    client._auth = YahooAuth(client._make_request_func())
    return client


# ---------------------------------------------------------------------------
# Basic get_json success
# ---------------------------------------------------------------------------


class TestClientGetJson:
    async def test_get_json_success(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("goodcrumb"))

        quote_data = {"quoteResponse": {"result": [{"symbol": "AAPL", "regularMarketPrice": 150}]}}
        handler.map_url(QUOTE_URL, FakeResponse(json_data=quote_data, is_json=True))

        client = make_client_with_handler(handler)
        result = await client.get_json(QUOTE_URL, params={"symbols": "AAPL"})

        assert result == quote_data

        # Verify crumb was injected
        quote_call = handler.find_call(QUOTE_URL)
        assert quote_call is not None
        assert quote_call["params"]["crumb"] == "goodcrumb"

    async def test_cookie_header_sent(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc; Path=/"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb123"))

        quote_data = {"quoteResponse": {"result": []}}
        handler.map_url(QUOTE_URL, FakeResponse(json_data=quote_data, is_json=True))

        client = make_client_with_handler(handler)
        await client.get_json(QUOTE_URL, params={"symbols": "AAPL"})

        quote_call = handler.find_call(QUOTE_URL)
        assert quote_call is not None
        assert quote_call["headers"] is not None
        assert "A1=abc" in quote_call["headers"]["Cookie"]


# ---------------------------------------------------------------------------
# 429 / Rate limit
# ---------------------------------------------------------------------------


class TestRateLimitHandling:
    async def test_429_raises_rate_limit_error(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb123"))
        handler.map_url(QUOTE_URL, FakeResponse(status_code=429, headers={"retry-after": "60"}))

        client = make_client_with_handler(handler)
        with pytest.raises(YahooRateLimitError) as exc_info:
            await client.get_json(QUOTE_URL)

        assert exc_info.value.retry_after == 60.0

    async def test_429_clears_crumb(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb123"))
        handler.map_url(QUOTE_URL, FakeResponse(status_code=429))

        client = make_client_with_handler(handler)
        route = YahooRoute()
        with pytest.raises(YahooRateLimitError):
            await client.get_json(QUOTE_URL, route=route)

        # Crumb should be cleared
        assert client._auth.get_crumb(route) is None

    async def test_429_without_retry_after(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb123"))
        handler.map_url(QUOTE_URL, FakeResponse(status_code=429))

        client = make_client_with_handler(handler)
        with pytest.raises(YahooRateLimitError) as exc_info:
            await client.get_json(QUOTE_URL)

        assert exc_info.value.retry_after is None


# ---------------------------------------------------------------------------
# Yahoo API error
# ---------------------------------------------------------------------------


class TestApiErrorHandling:
    async def test_yahoo_error_raises_typed_exception(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb123"))

        error_payload = {
            "finance": {"error": {"code": "Internal Error", "description": "Something broke"}}
        }
        handler.map_url(QUOTE_URL, FakeResponse(json_data=error_payload, is_json=True))

        client = make_client_with_handler(handler)
        with pytest.raises(YahooApiError, match="Internal Error"):
            await client.get_json(QUOTE_URL)


# ---------------------------------------------------------------------------
# Crumb error → strategy switch + retry
# ---------------------------------------------------------------------------


class TestCrumbRetrySwitch:
    async def test_crumb_error_triggers_strategy_switch(self) -> None:
        handler = FakeRequestHandler()
        # Basic strategy
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=basic"))
        handler.map_url(GETCRUMB_Q1, make_text_response("basiccrumb"))

        # First quote request returns crumb error
        crumb_error = {
            "finance": {"error": {"code": "Crumb validator failed", "description": "Invalid"}}
        }
        # Second pass: CSRF auth
        csrf_consent_html = (
            '<form><input type="hidden" name="sessionId" value="session-1"/>'
            '<input type="hidden" name="csrfToken" value="token-1"/></form>'
        )
        handler.map_url(GUCE_CONSENT, make_text_response(csrf_consent_html, cookie="GUC=x"))
        handler.map_url(CONSENT_COLLECT, make_text_response("", cookie="A1=posted"))
        handler.map_url(COPYCONSENT, make_text_response("", cookie="A1=csrf"))
        handler.map_url(GETCRUMB_Q2, make_text_response("csrfcrumb"))

        # Need to queue the crumb error first, then success
        handler.queue_response(FakeResponse(json_data=crumb_error, is_json=True))
        success_data = {"quoteResponse": {"result": [{"symbol": "AAPL"}]}}
        handler.queue_response(FakeResponse(json_data=success_data, is_json=True))

        client = make_client_with_handler(handler)
        result = await client.get_json(QUOTE_URL)

        assert result == success_data


# ---------------------------------------------------------------------------
# Proxy route isolation at client level
# ---------------------------------------------------------------------------


class TestProxyRouteIsolation:
    def test_configured_proxies_rotate_by_route(self) -> None:
        client = make_client_with_handler(FakeRequestHandler())
        client._proxies = ["http://proxy-a:8080", "http://proxy-b:8080"]

        assert client.get_route().proxy == "http://proxy-a:8080"
        assert client.get_route().proxy == "http://proxy-b:8080"
        assert client.get_route().proxy == "http://proxy-a:8080"

    async def test_different_routes_get_different_states(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb"))

        client = make_client_with_handler(handler)

        route_direct = client.get_route()
        route_proxy = client.get_route("http://proxy:8080")

        await client._auth.ensure_auth(route_direct)
        await client._auth.ensure_auth(route_proxy)

        state_direct = client._auth.get_state(route_direct)
        state_proxy = client._auth.get_state(route_proxy)

        assert state_direct is not state_proxy
        assert state_direct.route.proxy == ""
        assert state_proxy.route.proxy == "http://proxy:8080"

    async def test_clearing_one_route_doesnt_affect_other(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(FC_YAHOO, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB_Q1, make_text_response("crumb"))

        client = make_client_with_handler(handler)

        route_direct = client.get_route()
        route_proxy = client.get_route("http://proxy:8080")

        await client._auth.ensure_auth(route_direct)
        await client._auth.ensure_auth(route_proxy)

        client._auth.clear_crumb(route_direct)

        assert client._auth.get_crumb(route_direct) is None
        assert client._auth.get_crumb(route_proxy) == "crumb"


# ---------------------------------------------------------------------------
# Client cleanup
# ---------------------------------------------------------------------------


class TestClientCleanup:
    async def test_close_does_not_raise(self) -> None:
        handler = FakeRequestHandler()
        client = make_client_with_handler(handler)
        await client.close()  # should not raise

    async def test_context_manager(self) -> None:
        handler = FakeRequestHandler()
        client = make_client_with_handler(handler)
        async with client:
            pass
        # Should not raise on exit
