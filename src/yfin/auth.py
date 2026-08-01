"""Yahoo cookie/crumb authentication.

Adopts the yahooquery flow (no yahooquery dependency):

1. **Warmup** — GET ``https://finance.yahoo.com`` with redirects. This seeds
   the session cookie jar (A1/A3 cookies); in most regions any consent wall
   is resolved by following the redirect chain.

2. **Crumb** — GET ``https://query2.finance.yahoo.com/v1/test/getcrumb``.
   On failure the client proceeds crumb-less: the v8 chart API works without
   a crumb, and getcrumb is frequently rate-limited (429).

3. **CSRF fallback** — when the basic warmup fails entirely, run the explicit
   consent flow (``guce.yahoo.com/consent`` form parse, POST
   ``consent.yahoo.com/v2/collectConsent``, ``copyConsent``) exactly once,
   then request the crumb from query2 again.

Each :class:`YahooSessionState` owns an independent route (direct or a specific
proxy URL). Cookie/crumb state is **never** shared between routes.
"""

from __future__ import annotations

import asyncio
import html
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from html.parser import HTMLParser
from typing import Any

from .exceptions import YahooConsentError, YahooCrumbError
from .models import YahooRoute

__all__ = [
    "AuthStrategy",
    "YahooSessionState",
    "YahooAuth",
]

# URLs used by the two strategies (mirrors yahooquery's session setup).
_WARMUP_URL = "https://finance.yahoo.com"
_GETCRUMB = "https://query2.finance.yahoo.com/v1/test/getcrumb"
_GUCE_CONSENT = "https://guce.yahoo.com/consent"
_GUCE_COPYCONSENT = "https://guce.yahoo.com/copyConsent"
_CONSENT_COLLECT = "https://consent.yahoo.com/v2/collectConsent"


class AuthStrategy(StrEnum):
    """Authentication strategy identifiers."""

    BASIC = "basic"
    CSRF = "csrf"


# ---------------------------------------------------------------------------
# Consent form parser (stdlib html.parser — no lxml)
# ---------------------------------------------------------------------------


class _ConsentFormParser(HTMLParser):
    """Extract hidden input name/value pairs from the consent form."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, str] = {}
        self._in_form = False
        self._form_action: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "form":
            self._in_form = True
            for k, v in attrs:
                if k == "action":
                    self._form_action = v
        elif tag == "input" and self._in_form:
            name = None
            value = None
            itype = None
            for k, v in attrs:
                if k == "name":
                    name = v
                elif k == "value":
                    value = html.unescape(v) if v else ""
                elif k == "type":
                    itype = v
            if name and (itype in (None, "hidden", "submit")):
                self.inputs[name] = value or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._in_form = False


def parse_consent_html(html_text: str) -> dict[str, str]:
    """Parse consent HTML and return required hidden form fields.

    Raises :class:`YahooConsentError` when no form inputs are found.
    """
    parser = _ConsentFormParser()
    parser.feed(html_text)
    parser.close()
    if not parser.inputs:
        raise YahooConsentError("No hidden form fields found in consent response")
    return parser.inputs


# ---------------------------------------------------------------------------
# Crumb validation
# ---------------------------------------------------------------------------

_CRUMB_MIN_LEN = 2


def validate_crumb(crumb: str) -> str:
    """Validate that a crumb is non-blank and not HTML.

    Raises :class:`YahooCrumbError` on blank, HTML, or suspicious values.
    """
    stripped = crumb.strip()
    if not stripped:
        raise YahooCrumbError("Yahoo returned a blank crumb")
    # Yahoo crumbs are short opaque tokens; HTML indicates a consent wall.
    if "<html" in stripped.lower() or "<!doctype" in stripped.lower():
        raise YahooCrumbError("Yahoo returned HTML instead of a crumb (consent wall)")
    # A JSON body is an error payload (e.g. {"finance": {"error": ...}}),
    # never a crumb — accepting it poisons every subsequent request.
    if stripped.startswith(("{", "[")):
        raise YahooCrumbError(f"Yahoo returned JSON instead of a crumb: {stripped[:80]!r}")
    if len(stripped) < _CRUMB_MIN_LEN:
        raise YahooCrumbError(f"Yahoo crumb too short: {stripped!r}")
    return stripped


# ---------------------------------------------------------------------------
# Session state (per-route)
# ---------------------------------------------------------------------------


@dataclass
class YahooSessionState:
    """Cookie/crumb state for a single route.

    Holds the cached cookie value, crumb, and the currently active strategy.
    State is scoped to exactly one :class:`YahooRoute` so a proxy-obtained
    crumb is never reused on a different route.
    """

    route: YahooRoute
    cookie: str | None = None
    crumb: str | None = None
    strategy: AuthStrategy = AuthStrategy.BASIC
    switched: bool = field(default=False, init=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    """Whether the strategy has already been switched once (only one switch)."""

    def clear_crumb(self) -> None:
        self.crumb = None

    def has_valid_state(self) -> bool:
        return self.crumb is not None

    def can_switch_strategy(self) -> bool:
        return not self.switched

    def switch_strategy(self) -> AuthStrategy:
        """Switch to the CSRF fallback strategy. Can only be called once."""
        if self.switched:
            return self.strategy
        self.strategy = AuthStrategy.CSRF
        self.switched = True
        self.clear_crumb()
        return self.strategy


# ---------------------------------------------------------------------------
# Request function type alias
# ---------------------------------------------------------------------------

# A request function takes (url, method, params, headers, follow_redirects)
# and returns a Response-like object with .status_code, .text, .headers, .url,
# and .cookies (dict[str, str]).
RequestParam = dict[str, Any]
HeadersParam = dict[str, str]
RequestFunc = Callable[..., Awaitable[Any]]


# ---------------------------------------------------------------------------
# YahooAuth
# ---------------------------------------------------------------------------


class YahooAuth:
    """Manage cookie/crumb acquisition per route using fastreq.

    The actual HTTP requests are delegated to a caller-supplied async function
    (``request_func``). This keeps ``YahooAuth`` testable with fakes/fixtures
    and decoupled from the specific fastreq backend used at runtime.
    """

    def __init__(
        self,
        request_func: RequestFunc,
        states: dict[YahooRoute, YahooSessionState] | None = None,
    ) -> None:
        self._request_func = request_func
        self._states: dict[YahooRoute, YahooSessionState] = states or {}

    def get_state(self, route: YahooRoute) -> YahooSessionState:
        """Return (creating if needed) the state for *route*."""
        if route not in self._states:
            self._states[route] = YahooSessionState(route=route)
        return self._states[route]

    def get_crumb(self, route: YahooRoute) -> str | None:
        return self.get_state(route).crumb

    def get_cookie_header(self, route: YahooRoute) -> str | None:
        cookie = self.get_state(route).cookie
        if cookie is None:
            return None
        return cookie

    def clear_crumb(self, route: YahooRoute) -> None:
        self.get_state(route).clear_crumb()

    def switch_strategy(self, route: YahooRoute) -> AuthStrategy:
        return self.get_state(route).switch_strategy()

    def can_switch_strategy(self, route: YahooRoute) -> bool:
        return self.get_state(route).can_switch_strategy()

    # -- Cookie/crumb acquisition ------------------------------------------------

    async def ensure_auth(self, route: YahooRoute) -> YahooSessionState:
        """Ensure the state for *route* has a valid cookie and crumb.

        Uses the current strategy; switches to CSRF fallback exactly once if
        the basic strategy fails. If all strategies fail to obtain a crumb,
        returns a state with ``crumb=None`` — Yahoo's chart API works without
        a crumb, so callers can still proceed.
        """
        state = self.get_state(route)

        async with state.lock:
            if state.has_valid_state():
                return state

            while True:
                try:
                    await self._authenticate(state, route)
                    return state
                except YahooCrumbError, Exception:
                    if state.can_switch_strategy():
                        state.switch_strategy()
                        continue
                    # All strategies exhausted — proceed without crumb.
                    # The chart API works without one; getcrumb is frequently
                    # rate-limited (429) and the crumb is optional for chart
                    # endpoints.
                    return state

    async def _authenticate(self, state: YahooSessionState, route: YahooRoute) -> None:
        if state.strategy == AuthStrategy.BASIC:
            await self._basic_auth(state, route)
        else:
            await self._csrf_auth(state, route)

    # -- Basic strategy ----------------------------------------------------------

    async def _basic_auth(self, state: YahooSessionState, route: YahooRoute) -> None:
        """Basic: warmup finance.yahoo.com for cookies, then query2 getcrumb."""
        resp = await self._request_func(
            _WARMUP_URL, method="GET", route=route, follow_redirects=True
        )
        cookie = _extract_cookie(resp)
        if cookie:
            state.cookie = cookie

        crumb_resp = await self._request_func(
            _GETCRUMB,
            method="GET",
            route=route,
            follow_redirects=True,
            cookie=state.cookie,
        )
        crumb = _extract_text(crumb_resp)
        state.crumb = validate_crumb(crumb)

    # -- CSRF fallback strategy --------------------------------------------------

    async def _csrf_auth(self, state: YahooSessionState, route: YahooRoute) -> None:
        """CSRF: consent flow then query2 getcrumb."""
        # Step 1: GET consent page
        consent_resp = await self._request_func(
            _GUCE_CONSENT, method="GET", route=route, follow_redirects=True
        )
        cookie = _extract_cookie(consent_resp)
        if cookie:
            state.cookie = cookie

        consent_html = _extract_text(consent_resp)
        form_fields = parse_consent_html(consent_html)
        session_id = form_fields.get("sessionId")
        csrf_token = form_fields.get("csrfToken")
        if not session_id or not csrf_token:
            raise YahooConsentError("Yahoo consent form is missing sessionId or csrfToken")

        # Step 2: POST consent (yahooquery's proven payload: "agree" is sent
        # twice; pre-encoded so duplicate keys survive every backend)
        consent_payload = urllib.parse.urlencode(
            [
                ("agree", "agree"),
                ("agree", "agree"),
                ("consentUUID", "default"),
                ("sessionId", session_id),
                ("csrfToken", csrf_token),
                ("originalDoneUrl", _WARMUP_URL),
                ("namespace", "yahoo"),
            ]
        )
        post_resp = await self._request_func(
            f"{_CONSENT_COLLECT}?sessionId={session_id}",
            method="POST",
            route=route,
            data=consent_payload,
            follow_redirects=True,
            cookie=state.cookie,
        )
        cookie2 = _extract_cookie(post_resp)
        if cookie2:
            state.cookie = cookie2

        # Step 3: GET copyConsent
        copy_resp = await self._request_func(
            f"{_GUCE_COPYCONSENT}?sessionId={session_id}",
            method="GET",
            route=route,
            follow_redirects=True,
            cookie=state.cookie,
        )
        cookie3 = _extract_cookie(copy_resp)
        if cookie3:
            state.cookie = cookie3

        # Step 4: GET crumb from query2
        crumb_resp = await self._request_func(
            _GETCRUMB,
            method="GET",
            route=route,
            follow_redirects=True,
            cookie=state.cookie,
        )
        crumb = _extract_text(crumb_resp)
        state.crumb = validate_crumb(crumb)


# ---------------------------------------------------------------------------
# Response helpers (duck-typed — work with fastreq NormalizedResponse or fakes)
# ---------------------------------------------------------------------------


def _extract_cookie(resp: Any) -> str | None:
    """Extract a Set-Cookie value from a response.

    fastreq's NormalizedResponse has ``headers`` (lowercased dict).
    Multiple cookies may be in one Set-Cookie or separate.
    """
    headers = getattr(resp, "headers", None) or {}
    # Try both set-cookie and set-cookie variants.
    set_cookie = headers.get("set-cookie")
    if set_cookie:
        # Take the first cookie's name=value.
        first = set_cookie.split(",")[0] if isinstance(set_cookie, str) else set_cookie[0]
        return _cookie_string_to_header(first)

    # Some response objects expose a .cookies dict.
    cookies = getattr(resp, "cookies", None)
    if isinstance(cookies, dict) and cookies:
        # Return "name=value; name2=value2" form
        return "; ".join(f"{k}={v}" for k, v in cookies.items())

    return None


def _cookie_string_to_header(cookie_str: str) -> str:
    """Convert a raw Set-Cookie value to a Cookie header value."""
    # Strip attributes like ; Path=/; HttpOnly
    pair = cookie_str.split(";")[0].strip()
    return pair


def _extract_text(resp: Any) -> str:
    """Extract text content from a response."""
    # fastreq NormalizedResponse has .text
    text = getattr(resp, "text", None)
    if text is not None:
        return text
    content = getattr(resp, "content", None)
    if content is not None:
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content)
    raise YahooCrumbError("Response has no text or content attribute")
