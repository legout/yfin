# Handle errors and rate limits

yfin raises typed exceptions for every recoverable failure mode. The full
hierarchy is documented in [Reference / Exceptions](../reference/exceptions.md);
this guide covers the practical cases you'll actually see.

## Catch the whole family

All yfin exceptions inherit from `yfin.YahooError`:

```python
import yfin

try:
    quotes = yfin.quotes(["AAPL"])
except yfin.YahooError as exc:
    # Any yfin failure: auth, rate limit, validation, API error, …
    raise
```

## Symbol validation

`YahooSymbolError` is raised **before** any network call when you pass an
empty list, an invalid character, or a symbol longer than 12 chars:

```python
try:
    quotes = yfin.quotes(["aapl", "msft!"])
except yfin.YahooSymbolError as exc:
    print(exc)  # 'msft!': invalid characters
```

## Parameter validation

`YahooValidationError` is raised for invalid interval/range combinations,
explicit dates combined with `period`, and similar caller mistakes:

```python
try:
    yfin.history(["AAPL"], start=date(2025, 1, 1), period="1y")
except yfin.YahooValidationError as exc:
    print(exc)  # Cannot specify both explicit dates and period/range
```

## HTTP 429 — rate limited

Yahoo returns `429 Too Many Requests` when it has throttled your IP. yfin
raises `YahooRateLimitError` and includes `retry_after` if Yahoo sent a
`Retry-After` header:

```python
import time
import yfin

for attempt in range(5):
    try:
        quotes = yfin.quotes(["AAPL"])
        break
    except yfin.YahooRateLimitError as exc:
        wait = exc.retry_after or (2**attempt)
        time.sleep(wait)
else:
    raise
```

The crumb is automatically cleared on a 429 so the next retry starts clean.

## Auth failures

There are three flavours, all inheriting from `YahooAuthError`:

- `YahooCrumbError` — Yahoo returned a blank, HTML, or JSON-shaped "crumb"
- `YahooConsentError` — the CSRF consent flow couldn't be parsed
- `YahooAuthError` — base class; only raised when no crumb can be obtained

On a crumb failure, yfin automatically tries the CSRF fallback strategy
exactly once. If that also fails, it proceeds without a crumb — the chart
endpoint works crumb-less. Quotes / fundamentals / summary will then raise.

## API errors from Yahoo

`YahooApiError` covers structured error payloads (the `{"finance": {"error":
{"code": ..., "description": ...}}}` shape Yahoo returns for unknown symbols
and similar):

```python
try:
    quotes = yfin.quotes(["BAD_SYMBOL"])
except yfin.YahooApiError as exc:
    print(exc.code, exc)  # None, "Yahoo API error for …: Invalid symbol"
```

Per-symbol failures inside a multi-symbol call are logged at WARNING and
skipped. You only get `YahooApiError` when **every** symbol failed.

## What's next

- [Reference / Exceptions](../reference/exceptions.md) lists every class.
- [Explanation / Authentication flow](../explanation/authentication-flow.md)
  walks through the CSRF fallback in detail.
