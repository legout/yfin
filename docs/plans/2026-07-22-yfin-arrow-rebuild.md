# yfin Arrow rebuild — Implementation Plan

> **Execution model:** Implement in the isolated worktree with Hermes using `--provider z-ai --model glm-5.2`; retain this document as the source of truth.

**Goal:** Replace the unimportable legacy yfin package with a compact Python-3.14 Yahoo Finance client built on fastreq, providing robust batch quotes and historical OHLCV as `pyarrow.Table`, plus optional Polars conversion.

**Architecture:** yfin owns Yahoo-specific contracts only: symbol chunking, endpoints, stateful cookies/crumbs, JSON validation, and Arrow normalization. fastreq owns HTTP sessions, proxy pools, retries, rate pacing, headers, redirects, TLS, and streams. `YahooSession` maintains one cookie/crumb state per route/proxy so Yahoo state never leaks between rotating proxies.

**Tech stack:** Python 3.14, uv, fastreq >=3.0, PyArrow, optional Polars, pytest, Ruff, ty. No pandas, numpy, yfinance, yahooquery, requests, pendulum, lxml, or private yfinance imports.

---

## Public API

```python
import yfin

quotes: pyarrow.Table = yfin.quotes(["AAPL", "MSFT"], fields=["regularMarketPrice"])
history: pyarrow.Table = yfin.history(
    ["AAPL", "MSFT"],
    start=date(2016, 1, 1),
    end=date(2026, 1, 1),
    interval="1d",
    events=("div", "split"),
)

# Async equivalents return the same Arrow tables.
quotes = await yfin.quotes_async(["AAPL", "MSFT"])
history = await yfin.history_async(["AAPL", "MSFT"], period="1y")

# Optional only; raises a clear extra-install error if Polars is absent.
frame = yfin.to_polars(history)
```

Arrow is canonical. Schemas are explicit and deterministic:

- History: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `adjusted_close`, `volume`, `dividend`, `split_ratio`, `currency`, `exchange_timezone`.
- Quotes: `symbol` plus requested Yahoo fields normalized from camelCase to snake_case. Result columns use the caller field order after `symbol`; unknown/missing values are null.

`history` uses Yahoo `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`. It is necessarily one request per symbol; concurrency/pacing remains under fastreq control. `quotes` uses Yahoo `https://query1.finance.yahoo.com/v7/finance/quote` with bounded comma-separated symbol chunks.

## Cookie and crumb design

The implementation adapts the observed yfinance algorithm but does not import yfinance:

1. Each `YahooSessionState` owns an independent fastreq client route (direct or a chosen proxy), cookie jar, crumb, and active strategy.
2. Basic strategy: request `https://fc.yahoo.com`, then `https://query1.finance.yahoo.com/v1/test/getcrumb`.
3. CSRF fallback: GET `https://guce.yahoo.com/consent`, parse only required hidden inputs with stdlib `html.parser`, POST consent, GET copy-consent, then request `https://query2.finance.yahoo.com/v1/test/getcrumb`.
4. Cache a valid crumb only in its route-specific state. On crumb-invalid errors/429, clear it, switch strategy once, and retry through fastreq's bounded retry policy.
5. Treat an HTML crumb, blank crumb, consent parsing failure, or `Too Many Requests` response as typed Yahoo errors; never silently return empty tables.

## Scope boundaries

Included: batch quotes, chart history, dividends/splits, adjusted close, proxy-aware state, typed errors, Arrow/optional Polars, async APIs, synchronous wrappers that fail clearly inside a running event loop, tests, docs, CI.

Excluded: quoteSummary, company fundamentals, symbol search/lookup, options, free proxies, DataFrames from pandas, direct use of yfinance/yahooquery, a compatibility reimplementation of every legacy module.

## TDD implementation sequence

### 1. Package foundation

1. Replace the legacy package metadata with `uv_build`, `requires-python = ">=3.14"`, PyArrow runtime dependency, optional `polars` extra, and dev group `pytest`, `pytest-asyncio`, `ruff`, `ty`.
2. Declare `fastreq>=3.0.0` for release builds. During local cross-repository development install fastreq from its isolated worktree via `uv run --with /home/volker/coding/worktrees/fastreq-httpx-niquests` rather than committing an absolute source path.
3. Remove legacy source modules wholesale after adding the replacement module layout:
   - `src/yfin/__init__.py`
   - `src/yfin/client.py`
   - `src/yfin/auth.py`
   - `src/yfin/models.py`
   - `src/yfin/quotes.py`
   - `src/yfin/history.py`
   - `src/yfin/arrow.py`
   - `src/yfin/exceptions.py`
4. Add typed package exports, version declaration, README, `.gitignore`, and CI.

Verification:

```bash
uv sync --all-groups
uv run python -c 'import yfin; print(yfin.__version__)'
```

### 2. Yahoo transport/authentication contracts

1. Write fixture-driven tests for the basic cookie/crumb sequence, CSRF fallback, invalid/HTML/429 crumb responses, and per-route state isolation.
2. Add `YahooRoute`, `YahooSessionState`, and `YahooAuth` using fastreq's explicit request and proxy APIs.
3. Implement `get_json()` that appends the valid crumb itself, detects Yahoo error payloads, retries a changed crumb strategy once, and otherwise raises a typed exception with endpoint/symbol context.
4. Ensure no dependency or import from yfinance, pandas, requests, or private API remains.

Verification:

```bash
uv run pytest tests/test_auth.py -q
uv run ty check src/yfin/auth.py
```

### 3. Batch quote provider

1. Write tests from saved Yahoo v7 quote JSON fixtures for chunk boundaries, field ordering, camel-to-snake normalization, missing symbols, null values, duplicate input symbols, and empty result behavior.
2. Implement validated symbol normalization and URL-safe batch chunking with a conservative default of 200 symbols and an explicit maximum URL-size guard.
3. Implement `quotes_async()` with bounded concurrent chunk requests through one `YahooClient`; implement `quotes()` only as a non-running-loop convenience wrapper.
4. Normalize payloads directly into a deterministic `pyarrow.Table`; add `to_polars(table)` as the optional conversion utility.

Verification:

```bash
uv run pytest tests/test_quotes.py -q
uv run python -c 'import yfin; assert yfin.quotes(["AAPL"]).schema.names[0] == "symbol"'
```

### 4. Chart history provider

1. Write v8 chart JSON fixtures covering normal bars, missing volumes, null OHLC values, adjusted close, dividends, splits, Yahoo error payloads, and a symbol with no data.
2. Implement date/period validation and chart parameter construction (`period1`/`period2` or `range`, `interval`, `events`, `includePrePost`). Forbid specifying both explicit date range and `period`.
3. Implement bounded per-symbol concurrent retrieval using the same Yahoo client, preserving input-symbol order in output.
4. Convert chart payloads into the declared Arrow history schema. Null source values remain null; never coerce missing price data to zero. Derive `split_ratio` from `numerator`/`denominator` or Yahoo's split-ratio value.
5. Implement `history_async()` and a synchronous wrapper with the same event-loop behavior as quotes.

Verification:

```bash
uv run pytest tests/test_history.py -q
uv run ty check src/yfin/history.py
```

### 5. Reliability, documentation, and release gates

1. Add tests for 429/Retry-After, transient connection error, auth strategy switch, structured errors, direct route, proxy rotation route isolation, and client cleanup.
2. Document endpoints as unofficial Yahoo endpoints, conservative pacing, explicit proxy policy, environment variables, Arrow schema, optional Polars install, and unsupported legacy APIs.
3. Add CI for 3.14, test fixtures, Ruff, ty, and package build.
4. Run complete quality gates and an opt-in live smoke test for `AAPL,MSFT`; the fixture suite must pass without network access.

Final verification:

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest -q
uv build
uv run --with /home/volker/coding/worktrees/fastreq-httpx-niquests python -c 'import yfin; print(yfin.__version__)'
```

## Commit boundaries

1. `refactor: establish Python 3.14 Arrow package foundation`
2. `feat: add Yahoo cookie and crumb state management`
3. `feat: add Arrow batch quote provider`
4. `feat: add Arrow chart history provider`
5. `docs: document yfin migration and data contracts`

## Release sequence

1. Finish and tag/publish fastreq 3.0.0, or make an internal package artifact available.
2. Change yfin's normal dependency to `fastreq>=3.0.0`.
3. Run yfin integration against the released fastreq artifact, then publish/tag yfin as the first clean post-rewrite release.

## Acceptance criteria

- All legacy yfin imports and untested modules are removed rather than kept as broken compatibility surfaces.
- `quotes_async` and `history_async` produce deterministic Arrow tables from fixtures.
- Polars conversion is optional and clear when unavailable.
- Auth/cookies/crumbs use a direct reimplementation of the verified two-strategy flow, with no yfinance dependency.
- A proxy route never reuses a crumb/cookie state created by another route.
- Normal test suite is hermetic; live Yahoo tests are explicit and opt-in.
- Full formatting, linting, type checks, tests, and builds pass on Python 3.14.
