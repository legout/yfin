# Architecture

yfin is a thin Python wrapper over four unofficial Yahoo Finance endpoints.
It is deliberately small: one client, one auth layer, four providers, and a
single utility module that builds the Arrow tables.

## Module map

```
yfin (package)
├── __init__.py     # re-exports the public API
├── exceptions.py   # typed exception hierarchy
├── models.py       # HISTORY_SCHEMA, Interval, Range, QuoteFields,
│                   # YahooRoute, normalize_symbols, camel_to_snake
├── arrow.py        # build_quote_table, build_history_table,
│                   # build_fundamentals_table, build_summary_table, to_polars
├── auth.py         # AuthStrategy, YahooSessionState, YahooAuth
├── client.py       # YahooClient
├── quotes.py       # quotes, quotes_async
├── history.py      # history, history_async
├── fundamentals.py # fundamentals, fundamentals_async, *_TYPES
└── summary.py      # quote_summary, *_async + convenience wrappers
```

## Dependency direction

```
exceptions.py  ──┐
models.py      ──┤
                 ├──►  arrow.py  ──►  pyarrow
                 ├──►  auth.py
client.py      ──┤
                 └──►  quotes.py  ──►  YahooClient
                       history.py ──►  YahooClient
                       fundamentals.py ──►  YahooClient
                       summary.py    ──►  YahooClient
```

- `arrow.py` is at the bottom of the import graph — every provider depends
  on it, and it depends only on `models.py`.
- `auth.py` and `client.py` know about each other but neither imports the
  providers.
- The four providers (`quotes`, `history`, `fundamentals`, `summary`) all
  follow the same shape: validate input, build params, fan out via
  `asyncio.gather`, parse + concatenate, return `pa.Table`.

## The `YahooClient` boundary

Every async function in yfin accepts an optional `client: YahooClient`.
The client owns:

- a fastreq session (browser TLS impersonation, concurrency, retries)
- a list of proxy URLs (used for explicit-route selection)
- a `YahooAuth` instance (cookie/crumb acquisition)

When you don't pass a client, a transient one is created and closed for
you. When you do pass one, it must stay alive until the awaitable resolves.

## Sync / async split

Every provider function has two siblings: the synchronous `quotes` (or
`history`, …) and its `_async` counterpart `quotes_async`. The sync wrapper:

1. Checks that no event loop is running (raises `RuntimeError` otherwise).
2. Calls `asyncio.run(provider_async(...))`.

This split is mechanical and predictable. There are no thread pools, no
background tasks, no hidden asyncio.run calls inside the async API.

## Per-symbol fan-out

`history`, `fundamentals`, and `quote_summary` all issue **one request per
symbol**. yfin fans them out with `asyncio.gather(..., return_exceptions=True)`
so a single bad symbol doesn't poison the whole call:

- Successful responses are concatenated.
- Failures are logged at WARNING and skipped.
- If **every** symbol fails, the call raises `YahooApiError` with the first
  three error messages.

`quotes` is the exception: it batches symbols into a single comma-separated
`symbols=` parameter and chunks only when the URL would exceed ~8 KB.

## Why two auth strategies

Yahoo started returning consent walls in some regions. The CSRF fallback
strategy (consent form parse → POST collectConsent → copyConsent → retry
getcrumb) is only invoked **once** and only after the basic strategy fails.
This is what makes yfin work in regions where the basic flow gets stuck on
a consent page. See [Authentication flow](authentication-flow.md).

## What's next

- [Arrow output design](arrow-output-design.md) — the table-building layer.
- [Authentication flow](authentication-flow.md) — `YahooAuth` in detail.
- [Unofficial endpoints and pacing](unofficial-endpoints-and-pacing.md) —
  the conservative behaviour that comes from supporting undocumented APIs.
