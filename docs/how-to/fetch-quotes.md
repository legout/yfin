# Fetch quotes

Use [`quotes`](../reference/api.md) (sync) or
[`quotes_async`](../reference/api.md) to pull batch market
quotes from Yahoo's `v7/finance/quote` endpoint.

## Pick which fields you want

Yahoo returns roughly 80 fields per symbol. Restrict the list to keep the
response small:

```python
import yfin

quotes = yfin.quotes(
    ["AAPL", "MSFT", "GOOGL"],
    fields=[
        "regularMarketPrice",
        "regularMarketVolume",
        "regularMarketChangePercent",
        "currency",
        "shortName",
    ],
)
```

The fields are camelCase strings — they are sent verbatim to Yahoo and then
converted to snake_case for the Arrow column names. The order you pass them
in is the order of the columns in the returned table.

## Let yfin decide the columns

Pass `fields=None` and yfin will materialise whatever Yahoo returned:

```python
quotes = yfin.quotes(["AAPL"])
print(quotes.column_names)
```

Columns are still deterministic — yfin deduplicates the keys Yahoo happened to
send back and converts them to snake_case.

## Async

```python
import asyncio
import yfin


async def main():
    quotes = await yfin.quotes_async(["AAPL", "MSFT", "GOOGL"])
    print(quotes.num_rows)


asyncio.run(main())
```

The async variant accepts the same parameters and produces an identical
schema.

## Inside a running event loop

The sync wrapper fails loudly inside an event loop — it raises
`RuntimeError`. Use the async variant instead:

```python
async def handler():
    # WRONG: RuntimeError
    # quotes = yfin.quotes(["AAPL"])
    # RIGHT:
    quotes = await yfin.quotes_async(["AAPL"])
    ...
```

See [Choose sync or async](choose-sync-or-async.md) for the full reasoning.

## Chunking very large symbol lists

Yahoo caps the URL length at ~8 KB. yfin chunks automatically and respects a
default ceiling of 200 symbols per request:

```python
quotes = yfin.quotes(
    ["AAPL", "MSFT", ...thousands more...],
    chunk_size=150,  # smaller chunks if your symbols are long
)
```

The final table is the concatenation of every chunk's results, so callers see
one `pyarrow.Table` regardless of how many underlying requests were issued.

## Adding progress reporting

For very large fetches, hook into yfin's progress plumbing:

```python
quotes = yfin.quotes(
    ["AAPL", "MSFT", "GOOGL"],
    progress="tqdm",  # or "rich"
)
```

Headless environments should use `progress_callback`:

```python
quotes = yfin.quotes(
    ["AAPL", "MSFT", "GOOGL"],
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)
```

Both progress options require their respective extra:

```bash
uv pip install 'yfin-client[progress-rich]'
uv pip install 'yfin-client[progress-tqdm]'
```

## What's next

- [Reference / Arrow schemas](../reference/arrow-schemas.md#quotes) documents
  every column type and the null-row convention.
- [Explanation / Arrow output design](../explanation/arrow-output-design.md)
  covers how yfin infers Arrow types from mixed Python values.
