# Unofficial endpoints and pacing

Every endpoint yfin talks to is **undocumented**. Yahoo does not publish
the v7 quote, v8 chart, v10 quoteSummary, or fundamentals-timeseries APIs
that yfin uses — they are reverse-engineered from the public Yahoo Finance
web frontend and from community observation.

## What this means for you

- Yahoo can change the endpoints, the response shapes, or the auth flow at
  any time and without notice.
- A specific symbol's response may include extra fields today that won't
  be there tomorrow, or vice versa.
- Rate limits are not documented. yfin is conservative but cannot
  guarantee availability.

These caveats are not specific to yfin — every Yahoo Finance client on PyPI
depends on the same unofficial endpoints.

## Why yfin still uses them

The official Yahoo Finance APIs (the ones behind
[developer.yahoo.com](https://developer.yahoo.com)) don't expose the data
yfin needs at the granularity callers want. End-of-day prices, fundamentals
timeseries, and quoteSummary-style data are not on the public-developer
menu. yfin uses the unofficial endpoints so it can return the table shape
callers actually need.

## Conservative pacing

yfin is written to be a good citizen:

- **One cookie + one crumb per route, cached.** No per-request re-warmup.
- **Default concurrency 10.** Configurable per client.
- **Per-route state isolation.** Crumbs are not shared across routes.
- **No background polling.** Every request comes from a caller-driven call.
- **No scraping.** yfin does not scrape, mirror, or proxy Yahoo pages.

These are the same conventions the rest of the unofficial ecosystem uses
(yahooquery, yfinance, …). Going faster usually means getting throttled.

## What to do if you hit a 429

[`YahooRateLimitError`](../reference/exceptions.md#yahooratelimiterror) is
the typed exception yfin raises on HTTP 429. It carries the parsed
`Retry-After` value (or `None` if Yahoo didn't send one). See
[Handle errors and rate limits](../how-to/handle-errors-and-rate-limits.md)
for a worked retry-loop example.

## What to do if Yahoo changes an endpoint

Open an issue on GitHub with:

1. The symbol(s) you were calling.
2. The full URL yfin constructed.
3. The response shape you observed.

yfin's parsers are tolerant of unknown fields (they're ignored) and
explicit about the fields they need (missing required fields raise
`YahooApiError` with a clear message).

## What's next

- [Handle errors and rate limits](../how-to/handle-errors-and-rate-limits.md)
  — caller-side recovery.
- [Authentication flow](authentication-flow.md) — the auth side of the same
  "Yahoo can change anything" problem.
