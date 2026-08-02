# How-to guides

How-to guides are **task-oriented**. Pick the recipe that matches what you're
trying to do; each one assumes you're already productive with Python and just
need the yfin-specific steps.

## Fetching data

- [Fetch quotes](fetch-quotes.md) — `quotes` / `quotes_async`, fields,
  chunking
- [Fetch history](fetch-history.md) — `history` / `history_async`, intervals,
  periods, events
- [Fetch fundamentals](fetch-fundamentals.md) — `fundamentals`, type groups,
  date ranges
- [Fetch summary modules](fetch-summary-modules.md) — `quote_summary`,
  `asset_profile`, `key_statistics`, etc.

## Configuring behaviour

- [Configure proxies](configure-proxies.md) — route isolation, per-proxy
  cookie/crumb state
- [Handle errors and rate limits](handle-errors-and-rate-limits.md) — the
  exception hierarchy and `429` retries
- [Choose sync or async](choose-sync-or-async.md) — when `asyncio.run` is
  safe, when to reach for the async API directly

If a guide doesn't match your question, look for a term in the
[Reference](../reference/index.md), or read about the underlying design in
[Explanation](../explanation/index.md).
