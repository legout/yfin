"""Tests for the Yahoo cookie/crumb authentication flow.

Covers: warmup + crumb (yahooquery-style basic strategy), CSRF consent
fallback, invalid/HTML/blank/JSON crumb responses, per-route state isolation,
and strategy switch (once only).
"""

from __future__ import annotations

import asyncio
import urllib.parse

import pytest

from yfin.auth import (
    AuthStrategy,
    YahooAuth,
    YahooSessionState,
    parse_consent_html,
    validate_crumb,
)
from yfin.exceptions import YahooConsentError, YahooCrumbError
from yfin.models import YahooRoute

from .conftest import (
    FakeRequestHandler,
    make_request_func,
    make_text_response,
)

# URLs used by yfin.auth (yahooquery-style flow)
WARMUP = "https://finance.yahoo.com"
GETCRUMB = "https://query2.finance.yahoo.com/v1/test/getcrumb"
GUCE_CONSENT = "https://guce.yahoo.com/consent"
CONSENT_COLLECT = "https://consent.yahoo.com/v2/collectConsent?sessionId=abc123"
COPYCONSENT = "https://guce.yahoo.com/copyConsent?sessionId=abc123"


# ---------------------------------------------------------------------------
# validate_crumb
# ---------------------------------------------------------------------------


class TestValidateCrumb:
    def test_valid_crumb(self) -> None:
        assert validate_crumb("abc123XYZ") == "abc123XYZ"

    def test_strips_whitespace(self) -> None:
        assert validate_crumb("  abc123  ") == "abc123"

    def test_blank_crumb_raises(self) -> None:
        with pytest.raises(YahooCrumbError, match="blank"):
            validate_crumb("   ")

    def test_html_crumb_raises(self) -> None:
        with pytest.raises(YahooCrumbError, match="HTML"):
            validate_crumb("<html><body>consent</body></html>")

    def test_doctype_crumb_raises(self) -> None:
        with pytest.raises(YahooCrumbError, match="HTML"):
            validate_crumb("<!DOCTYPE html>")

    def test_json_error_payload_raises(self) -> None:
        """A JSON error body is never a crumb (crumb poisoning regression)."""
        payload = '{"finance":{"result":null,"error":{"code":"Unauthorized"}}}'
        with pytest.raises(YahooCrumbError, match="JSON"):
            validate_crumb(payload)

    def test_json_array_raises(self) -> None:
        with pytest.raises(YahooCrumbError, match="JSON"):
            validate_crumb("[]")

    def test_too_short_crumb_raises(self) -> None:
        with pytest.raises(YahooCrumbError, match="too short"):
            validate_crumb("a")


# ---------------------------------------------------------------------------
# parse_consent_html
# ---------------------------------------------------------------------------

CONSENT_HTML = """
<html><body>
<form action="/consent" method="post">
    <input type="hidden" name="sessionId" value="abc123" />
    <input type="hidden" name="csrfToken" value="tokXYZ" />
    <input type="submit" name="agree" value="Agree" />
    <input type="text" name="ignored_field" value="data" />
</form>
</body></html>
"""


class TestParseConsentHtml:
    def test_extracts_hidden_fields(self) -> None:
        fields = parse_consent_html(CONSENT_HTML)
        assert fields["sessionId"] == "abc123"
        assert fields["csrfToken"] == "tokXYZ"
        assert fields["agree"] == "Agree"

    def test_no_form_raises(self) -> None:
        with pytest.raises(YahooConsentError, match="No hidden form fields"):
            parse_consent_html("<html><body>No form here</body></html>")

    def test_empty_html_raises(self) -> None:
        with pytest.raises(YahooConsentError, match="No hidden form fields"):
            parse_consent_html("")

    def test_html_entities_decoded(self) -> None:
        html_text = '<form><input type="hidden" name="val" value="a&amp;b"/></form>'
        fields = parse_consent_html(html_text)
        assert fields["val"] == "a&b"


# ---------------------------------------------------------------------------
# YahooSessionState
# ---------------------------------------------------------------------------


class TestYahooSessionState:
    def test_default_strategy_is_basic(self) -> None:
        state = YahooSessionState(route=YahooRoute())
        assert state.strategy == AuthStrategy.BASIC
        assert not state.has_valid_state()

    def test_clear_crumb(self) -> None:
        state = YahooSessionState(route=YahooRoute(), crumb="abc")
        assert state.has_valid_state()
        state.clear_crumb()
        assert not state.has_valid_state()

    def test_switch_strategy_once(self) -> None:
        state = YahooSessionState(route=YahooRoute())
        assert state.can_switch_strategy()
        new_strategy = state.switch_strategy()
        assert new_strategy == AuthStrategy.CSRF
        assert state.switched
        assert not state.can_switch_strategy()

    def test_switch_strategy_twice_returns_csrf(self) -> None:
        state = YahooSessionState(route=YahooRoute())
        state.switch_strategy()
        result = state.switch_strategy()
        assert result == AuthStrategy.CSRF  # stays CSRF


# ---------------------------------------------------------------------------
# YahooAuth: basic strategy (warmup + query2 getcrumb)
# ---------------------------------------------------------------------------


class TestBasicAuth:
    async def test_basic_strategy_success(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=abc; Path=/; HttpOnly"))
        handler.map_url(GETCRUMB, make_text_response("validcrumb123"))

        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        state = await auth.ensure_auth(route)

        assert state.strategy == AuthStrategy.BASIC
        assert state.crumb == "validcrumb123"
        assert state.cookie is not None
        assert "A1=abc" in state.cookie

    async def test_cached_state_not_reauthenticated(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB, make_text_response("crumbXYZ"))

        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        await auth.ensure_auth(route)
        initial_call_count = handler.call_count

        # Second call should not re-authenticate
        await auth.ensure_auth(route)
        assert handler.call_count == initial_call_count

    async def test_concurrent_auth_for_one_route_uses_one_cookie_flow(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB, make_text_response("crumbXYZ"))
        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        first, second = await asyncio.gather(auth.ensure_auth(route), auth.ensure_auth(route))

        assert first is second
        assert handler.url_call_count(WARMUP) == 1
        assert handler.url_call_count(GETCRUMB) == 1

    async def test_basic_blank_crumb_switches_to_csrf(self) -> None:
        handler = FakeRequestHandler()
        # Basic strategy: warmup ok, blank crumb -> switch
        handler.map_url(WARMUP, make_text_response("", cookie="A1=abc"))

        # CSRF strategy responses (consent flow)
        handler.map_url(GUCE_CONSENT, make_text_response(CONSENT_HTML, cookie="GUC=xyz"))
        handler.map_url(CONSENT_COLLECT, make_text_response("", cookie="A1=posted"))
        handler.map_url(COPYCONSENT, make_text_response("", cookie="A1=def"))

        # Both strategies hit the same getcrumb URL — queue responses in
        # call order: basic crumb (blank), then CSRF crumb (good).
        handler.queue_response(make_text_response("   "))
        handler.queue_response(make_text_response("csrfcrumb456"))

        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        state = await auth.ensure_auth(route)

        assert state.strategy == AuthStrategy.CSRF
        assert state.switched
        assert state.crumb == "csrfcrumb456"
        post_call = handler.find_call(CONSENT_COLLECT)
        assert post_call is not None
        # Consent payload is a pre-encoded form string (yahooquery style,
        # with "agree" sent twice)
        pairs = urllib.parse.parse_qsl(post_call["data"])
        assert ("csrfToken", "tokXYZ") in pairs
        assert ("sessionId", "abc123") in pairs
        assert pairs.count(("agree", "agree")) == 2
        assert ("namespace", "yahoo") in pairs


# ---------------------------------------------------------------------------
# YahooAuth: CSRF strategy fallback
# ---------------------------------------------------------------------------


class TestCsrfAuth:
    async def test_csrf_strategy_success(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=basic"))
        handler.map_url(GUCE_CONSENT, make_text_response(CONSENT_HTML, cookie="GUC=consent"))
        handler.map_url(CONSENT_COLLECT, make_text_response("", cookie="A1=posted"))
        handler.map_url(COPYCONSENT, make_text_response("", cookie="A1=csrf"))

        # basic crumb is an HTML consent wall -> switch; CSRF crumb is good
        handler.queue_response(make_text_response("<html>consent wall</html>"))
        handler.queue_response(make_text_response("goodcrumb789"))

        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        state = await auth.ensure_auth(route)
        assert state.strategy == AuthStrategy.CSRF
        assert state.crumb == "goodcrumb789"

    async def test_csrf_consent_failure_when_no_form(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=basic"))
        handler.map_url(GUCE_CONSENT, make_text_response("no form here"))

        # basic crumb blank -> switch; consent parse fails
        handler.queue_response(make_text_response(""))

        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        # The consent parse fails; with both strategies exhausted, ensure_auth
        # proceeds crumb-less (crumb is optional for the chart API).
        state = await auth.ensure_auth(route)
        assert state.crumb is None
        assert state.strategy == AuthStrategy.CSRF


# ---------------------------------------------------------------------------
# YahooAuth: strategy switch (once only)
# ---------------------------------------------------------------------------


class TestStrategySwitch:
    async def test_switch_only_once(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=basic"))
        # CSRF consent flow succeeds mechanically...
        handler.map_url(GUCE_CONSENT, make_text_response(CONSENT_HTML, cookie="GUC=xyz"))
        handler.map_url(CONSENT_COLLECT, make_text_response("", cookie="A1=posted"))
        handler.map_url(COPYCONSENT, make_text_response("", cookie="A1=csrf"))
        # ...but both crumb attempts come back blank
        handler.queue_response(make_text_response("  "))
        handler.queue_response(make_text_response("  "))

        auth = YahooAuth(make_request_func(handler))
        route = YahooRoute()

        # Both strategies yield blank crumbs: switch happens exactly once,
        # then ensure_auth proceeds crumb-less instead of raising.
        state = await auth.ensure_auth(route)
        assert state.crumb is None
        assert state.strategy == AuthStrategy.CSRF
        assert state.switched is True


# ---------------------------------------------------------------------------
# YahooAuth: per-route state isolation
# ---------------------------------------------------------------------------


class TestRouteIsolation:
    async def test_direct_and_proxy_have_separate_state(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=direct"))
        handler.map_url(GETCRUMB, make_text_response("direct_crumb"))

        auth = YahooAuth(make_request_func(handler))
        direct = YahooRoute(proxy="")
        proxy1 = YahooRoute(proxy="http://proxy1:8080")

        direct_state = await auth.ensure_auth(direct)
        proxy1_state = await auth.ensure_auth(proxy1)

        assert direct_state.crumb == "direct_crumb"
        assert proxy1_state.crumb == "direct_crumb"  # same handler, but separate state obj
        assert direct_state is not proxy1_state
        assert direct_state.route != proxy1_state.route

    async def test_two_proxies_have_separate_states(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB, make_text_response("shared_crumb"))

        auth = YahooAuth(make_request_func(handler))
        p1 = YahooRoute(proxy="http://proxy1:8080")
        p2 = YahooRoute(proxy="http://proxy2:8080")

        s1 = await auth.ensure_auth(p1)
        s2 = await auth.ensure_auth(p2)

        assert s1 is not s2
        assert s1.route.proxy == "http://proxy1:8080"
        assert s2.route.proxy == "http://proxy2:8080"

    async def test_clearing_one_route_does_not_affect_other(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=abc"))
        handler.map_url(GETCRUMB, make_text_response("crumb"))

        auth = YahooAuth(make_request_func(handler))
        p1 = YahooRoute(proxy="http://proxy1:8080")
        p2 = YahooRoute(proxy="http://proxy2:8080")

        s1 = await auth.ensure_auth(p1)
        await auth.ensure_auth(p2)

        auth.clear_crumb(p1)
        assert s1.crumb is None
        # p2 should still be valid
        assert auth.get_crumb(p2) == "crumb"


# ---------------------------------------------------------------------------
# YahooAuth: cookie header extraction
# ---------------------------------------------------------------------------


class TestCookieExtraction:
    async def test_cookie_header_passed_to_crumb_request(self) -> None:
        handler = FakeRequestHandler()
        handler.map_url(WARMUP, make_text_response("", cookie="A1=testcookie; Path=/"))
        handler.map_url(GETCRUMB, make_text_response("crumb123"))

        auth = YahooAuth(make_request_func(handler))
        await auth.ensure_auth(YahooRoute())

        crumb_call = handler.find_call(GETCRUMB)
        assert crumb_call is not None
        # The cookie should have been passed
        assert crumb_call.get("cookie") is not None
        assert "A1=testcookie" in crumb_call["cookie"]
