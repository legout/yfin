# Configure proxies

yfin uses [fastreq](https://github.com/legout/fastreq) under the hood, which
means proxy configuration goes through fastreq's explicit-proxy transport.
The yfin layer adds one thing on top: **per-route cookie/crumb isolation** so
that a Yahoo cookie obtained through proxy A is never sent through proxy B.

## Pass a list of proxies

```python
import yfin

client = yfin.YahooClient(
    proxies=["http://proxy1.example:8080", "http://proxy2.example:8080"],
)

quotes = yfin.quotes(["AAPL", "MSFT"], client=client)
```

yfin assigns routes round-robin. Each route — direct or one specific proxy —
maintains its own cookie/crumb cache.

## Pin a request to a specific proxy

Pass `proxy=...` to pin a single call to a single route:

```python
quotes = await yfin.quotes_async(
    ["AAPL"],
    client=client,
    proxy="http://proxy1.example:8080",
)
```

The state obtained for that proxy is reused across calls but never leaked to
the direct route or to another proxy.

## Why explicit-proxy, not transport-level proxy switching

fastreq's default behaviour picks a proxy per request from a pool. yfin does
**not** rely on that — it selects a route up front and forwards the proxy
URL explicitly to fastreq. The reason: Yahoo's auth flow expects a stable
session per IP. A silent transport-level switch would let a cookie obtained
through one IP be sent through another, which Yahoo rejects.

## Reuse one client for many calls

A `YahooClient` is async. Use it as a context manager for a single async
session:

```python
async with yfin.YahooClient(proxies=[...]) as client:
    quotes = await yfin.quotes_async(["AAPL", "MSFT"], client=client)
    history = await yfin.history_async(["AAPL"], period="1y", client=client)
```

Or close it manually:

```python
client = yfin.YahooClient(proxies=[...])
try:
    quotes = await yfin.quotes_async(["AAPL"], client=client)
finally:
    await client.close()
```

When you don't pass `client=`, yfin creates a transient client per call and
closes it for you. Don't reuse a closed client.

## Free proxies

yfin does not fetch, rotate, or scrape free proxies. Configure them yourself
— the [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html)
restrict automated access regardless of how the requests are routed.

## What's next

- [Explanation / Architecture](../explanation/architecture.md) for the full
  module map (how `YahooClient`, `YahooAuth`, and the providers fit together).
- [Handle errors and rate limits](handle-errors-and-rate-limits.md) for what
  happens when a proxy gets throttled.
