# Convert to Polars

yfin's primary output is `pyarrow.Table`. Polars is a popular alternative
DataFrame library that interoperates cleanly with Arrow — and yfin ships a
helper to make the conversion one call.

## 1. Install the extra

```bash
uv pip install 'yfin-client[polars]'
```

This pulls in `polars>=1.0.0` alongside the core dependencies. The extra is
optional because not every yfin user wants Polars as a runtime dependency.

## 2. Convert any table

`yfin.to_polars` accepts any `pyarrow.Table` returned by yfin:

```python
import yfin

quotes = yfin.quotes(["AAPL", "MSFT"])
df = yfin.to_polars(quotes)

print(type(df).__name__)  # DataFrame
print(df.schema)
```

The conversion is lossless: every Arrow column becomes a Polars column with a
matching type, and the row order is preserved.

## 3. Use it in a chain

The helper returns a `polars.DataFrame`, so you can keep going with Polars
expressions:

```python
import yfin

quotes = yfin.to_polars(yfin.quotes(["AAPL", "MSFT", "GOOGL", "AMZN"]))

big_movers = (
    quotes.filter(pl.col("regular_market_change_percent").abs() > 2.0)
    .select(["symbol", "regular_market_price", "regular_market_change_percent"])
    .sort("regular_market_change_percent", descending=True)
)
```

## 4. What you give up

- One import: every script that touches `to_polars` needs the `polars` package
  to be importable. If you skip the extra, you'll get a clear `ImportError`
  pointing at the install command.
- A copy: `pyarrow.Table` and `polars.DataFrame` share the underlying buffers
  in some conversions but not all. Treat the result as a new value.

## What's next?

- [Reference / API](../reference/api.md) lists every public symbol yfin
  exports, including `to_polars`.
- [Explanation / Arrow output design](../explanation/arrow-output-design.md)
  covers why yfin settled on `pyarrow.Table` as the primary return type.
