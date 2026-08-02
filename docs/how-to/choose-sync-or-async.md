# Choose sync or async

Every public function in yfin has two flavours: a sync wrapper
(`quotes`, `history`, `fundamentals`, `quote_summary`, `asset_profile`, …)
and an `_async` counterpart (`quotes_async`, `history_async`, …). Pick the
one that matches your call site.

## Use the sync wrapper when…

You're in a script, a Jupyter cell, a CLI, or any other context where there's
no running event loop:

```python
import yfin

quotes = yfin.quotes(["AAPL"])
history = yfin.history(["AAPL"], period="1y")
```

The sync wrapper calls `asyncio.run(...)` internally. It is straightforward
and the obvious default for one-off fetches.

## Use the async API when…

You're inside a coroutine, a FastAPI/HTTP handler, a notebook already running
an event loop, or any other context where `asyncio.run` would fail:

```python
async def fetch():
    quotes = await yfin.quotes_async(["AAPL"])
    history = await yfin.history_async(["AAPL"], period="1y")
    return quotes, history
```

Reusing one `YahooClient` across many concurrent fetches is materially faster
because the cookie/crumb is acquired once and shared.

## What happens if you mix them up

The sync wrappers detect a running event loop and raise `RuntimeError`
instead of deadlocking:

```python
async def bad():
    yfin.quotes(["AAPL"])  # RuntimeError: yfin sync wrappers must not be
    # called from a running event loop. Use the async
    # variant (quotes_async) instead.
```

If you see this error, the fix is to use the `_async` counterpart.

## Reusing one client for many fetches

Sync callers can't reuse a client across calls (the sync wrapper creates a
transient one). Async callers can:

```python
import yfin


async def main():
    async with yfin.YahooClient() as client:
        quotes = await yfin.quotes_async(["AAPL"], client=client)
        history = await yfin.history_async(["AAPL"], client=client, period="1y")
        fundamentals = await yfin.fundamentals_async(
            ["AAPL"],
            client=client,
            types=yfin.VALUATION_TYPES,
        )
```

## What's next

- [Configure proxies](configure-proxies.md) for proxy pool configuration,
  which only makes sense in the async case.
- [Explanation / Architecture](../explanation/architecture.md) for the full
  module map.
