# Arrow output design

yfin returns `pyarrow.Table` from every public function. This page explains
why, and walks through how the column types are decided.

## Why Arrow

- **Zero-copy interop** with Polars, DuckDB, pandas 2.x, and most modern
  data tools.
- **Schema is part of the value.** A `pa.Table` knows its column types, so
  callers don't have to inspect the JSON to discover them.
- **Deterministic schemas.** Every `yfin.history` call returns the same
  twelve columns in the same order, regardless of which Yahoo fields
  happened to be populated.

The Polars extra is offered as a thin convenience (`yfin.to_polars`) but
isn't required — callers who only need Arrow don't pay the import cost.

## The `build_*_table` functions

Every provider hands its raw Yahoo JSON to a dedicated builder in
`yfin.arrow`:

- `build_quote_table` — one row per symbol, columns from caller-provided
  fields
- `build_history_table` — one row per OHLCV bar, schema pinned to
  `HISTORY_SCHEMA`
- `build_fundamentals_table` — one row per `(symbol, as_of_date)` pair
- `build_summary_table` — flat rows + event rows concatenated

Each builder is pure: same input, same output. No hidden globals, no I/O.

## Type inference (quotes)

Yahoo's quote responses are loosely typed. A `regularMarketPrice` might
come back as a number or as a `{"raw": ...}` wrapper. yfin's
`_infer_arrow_array` helper uses this rule:

1. If every non-null value is `bool` → `bool_`
2. Else if every non-null value is `int` (not bool) → `int64`
3. Else if every non-null value is `int` or `float` (not bool) → `float64`
4. Otherwise → `string`
5. All-null → `string` (safe default)

This rule is intentionally simple. It avoids guessing wrong when a column
is mostly numeric with one stray string, and it always produces a column
of the same type across symbols (the inferred type is fixed once we see
the first non-null value).

## Canonical schemas (history)

`HISTORY_SCHEMA` is the only schema that matters for the chart endpoint
— every call returns the same twelve columns in the same order. The
`build_history_table` helper pads and casts Yahoo's slightly-less-tidy
output into the canonical shape, so callers never have to worry about a
row landing in a different column from one call to the next.

## Wide columns (summary)

The summary builder flattens Yahoo's nested JSON into wide columns:

- Module name is the prefix: `assetProfile.industry` →
  `asset_profile_industry` (after the dot-to-underscore replacement).
- Yahoo's `{"raw": X}` wrappers are unwrapped recursively.
- Calendar events and flat profiles produce one row per symbol.
- History/ownership modules produce one row per event.

Flat and event rows are concatenated row-wise with null padding on the
side that doesn't apply. This lets you concatenate them with other yfin
outputs and feed them straight into a Polars pipeline.

## The null-row convention

If you ask for `["AAPL", "MISSING_SYMBOL"]`, you get **two rows**: the
AAPL one populated and the MISSING one with nulls in every column. yfin
does **not** silently drop the missing row. This makes it easy to spot
bad symbols downstream and to keep your join keys stable.

## What's next

- [Reference / Arrow schemas](../reference/arrow-schemas.md) — every
  schema in tabular form.
- [Architecture](architecture.md) — the layer that calls these builders.
