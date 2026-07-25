"""Yahoo client: integrates fastreq with yfin auth.

The :class:`YahooClient` owns a fastreq session and a :class:`YahooAuth`
instance. It exposes ``get_json()`` which handles cookie/crumb injection,
retry-once-on-crumb-failure, and typed error raising.
"""

from __future__ import annotations

import json
from typing import Any

import fastreq

from .auth import RequestFunc, YahooAuth
from .exceptions import (
    YahooApiError,
    YahooCrumbError,
    YahooError,
    YahooRateLimitError,
)
from .models import YahooRoute, detect_yahoo_error

__all__ = ["YahooClient"]

# yahooquery-style default query parameters sent with every API request.
_DEFAULT_QUERY_PARAMS: dict[str, str] = {
    "lang": "en-US",
    "region": "US",
    "corsDomain": "finance.yahoo.com",
}


class YahooClient:
    """High-level Yahoo client wrapping a fastreq session.

    Parameters
    ----------
    proxies
        Optional list of proxy URLs. When provided, each proxy gets its own
        route/state in :class:`YahooAuth`.
    concurrency
        Maximum concurrent requests through fastreq.
    max_retries
        Number of retry attempts per HTTP request (handled by fastreq).
    rate_limit
        Requests per second cap (None for no limit).
    **kwargs
        Additional keyword arguments forwarded to fastreq.
    """

    def __init__(
        self,
        proxies: list[str] | None = None,
        *,
        concurrency: int = 10,
        max_retries: int = 3,
        rate_limit: float | None = None,
        backend: str = "curl_cffi",
        impersonate: str | None = "random",
        timeout: float | None = 10.0,
        follow_redirects: bool = True,
        random_user_agent: bool = True,
        headers: dict[str, str] | None = None,
        **fastreq_kwargs: Any,
    ) -> None:
        self._fastreq = fastreq.FastRequests(
            backend=backend,
            concurrency=concurrency,
            max_retries=max_retries,
            rate_limit=rate_limit,
            impersonate=impersonate,
            timeout=timeout,
            follow_redirects=follow_redirects,
            random_user_agent=random_user_agent,
            headers=headers,
            proxies=proxies,
            **fastreq_kwargs,
        )
        # Store proxy list for route selection
        self._proxies = list(proxies) if proxies else []
        self._proxy_index = 0

        self._auth = YahooAuth(self._make_request_func())

    def _make_request_func(self) -> RequestFunc:
        """Create the async request function used by YahooAuth."""

        async def _request(
            url: str,
            method: str = "GET",
            route: YahooRoute | None = None,
            data: dict[str, str] | None = None,
            follow_redirects: bool | None = None,
            cookie: str | None = None,
            **_kwargs: Any,
        ) -> Any:
            headers: dict[str, str] = {}
            if cookie:
                headers["Cookie"] = cookie
            kwargs: dict[str, Any] = {
                "method": method,
                "headers": headers or None,
                "follow_redirects": follow_redirects if follow_redirects is not None else True,
                "return_type": "response",
            }
            if data is not None:
                kwargs["data"] = data
            if route and route.proxy:
                kwargs["proxy"] = route.proxy

            return await self._fastreq.request(url, **kwargs)

        return _request

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        """Return an explicit, route-stable proxy or the direct route.

        Selecting a proxy here and forwarding it explicitly to fastreq keeps a
        Yahoo cookie/crumb bound to its actual network route. A hidden
        transport-level proxy switch would make that state unsafe to reuse.
        """
        if proxy is not None:
            return YahooRoute(proxy=proxy)
        if not self._proxies:
            return YahooRoute()
        selected = self._proxies[self._proxy_index % len(self._proxies)]
        self._proxy_index += 1
        return YahooRoute(proxy=selected)

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        route: YahooRoute | None = None,
    ) -> Any:
        """Fetch JSON from *url* with cookie/crumb injection.

        Handles:
        - Appending the crumb to query params.
        - Switching auth strategy once on crumb failure.
        - Detecting Yahoo error payloads and raising typed exceptions.
        - HTTP 429 → :class:`YahooRateLimitError`.
        """
        if route is None:
            route = YahooRoute()

        state = await self._auth.ensure_auth(route)
        crumb = state.crumb

        request_params = {**_DEFAULT_QUERY_PARAMS, **(params or {})}
        if crumb:
            request_params["crumb"] = crumb

        headers: dict[str, str] = {}
        cookie = self._auth.get_cookie_header(route)
        if cookie:
            headers["Cookie"] = cookie

        fr_kwargs: dict[str, Any] = {
            "method": "GET",
            "params": request_params,
            "headers": headers or None,
            "return_type": "response",
        }
        if route.proxy:
            fr_kwargs["proxy"] = route.proxy

        resp = await self._fastreq.request(url, **fr_kwargs)

        # Check for 429
        status = getattr(resp, "status_code", 200)
        if status == 429:
            retry_after = _parse_retry_after(resp)
            self._auth.clear_crumb(route)
            raise YahooRateLimitError(f"Yahoo returned 429 for {url}", retry_after=retry_after)

        # Parse JSON from response
        json_data = _get_json(resp)

        # Check for Yahoo API errors
        err_msg = detect_yahoo_error(json_data)
        if err_msg:
            # If it's a crumb error and we haven't switched yet, retry once.
            is_crumb_error = "crumb" in err_msg.lower()
            if is_crumb_error and self._auth.can_switch_strategy(route):
                self._auth.clear_crumb(route)
                self._auth.switch_strategy(route)
                return await self.get_json(url, params=params, route=route)
            raise YahooApiError(f"Yahoo API error for {url}: {err_msg}")

        return json_data

    async def close(self) -> None:
        """Close the underlying fastreq session."""
        await self._fastreq.close()

    async def __aenter__(self) -> YahooClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.close()


def _parse_retry_after(resp: Any) -> float | None:
    """Parse Retry-After header from a response."""
    headers = getattr(resp, "headers", None) or {}
    ra = headers.get("retry-after")
    if ra is None:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


def _get_json(resp: Any) -> Any:
    """Extract JSON data from a fastreq NormalizedResponse or fake."""
    # fastreq NormalizedResponse has .json_data
    json_data = getattr(resp, "json_data", _MISSING)
    if json_data is not _MISSING and json_data is not None:
        return json_data
    # Try parsing from text.
    text = getattr(resp, "text", None)
    if text is not None:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise YahooCrumbError(f"Yahoo returned non-JSON response: {text[:200]}") from exc
    raise YahooError("Response has no JSON data")


_MISSING: Any = object()
