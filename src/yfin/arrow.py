"""Arrow table construction for yfin.

These functions convert validated Yahoo JSON payloads into deterministic
``pyarrow.Table`` objects whose schemas are documented in :mod:`yfin.models`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from .models import HISTORY_SCHEMA, camel_to_snake

__all__ = [
    "build_quote_table",
    "build_history_table",
    "to_polars",
]


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

_MISSING = object()


def build_quote_table(
    quotes_data: list[dict[str, Any]],
    *,
    fields: Sequence[str] | None,
    requested_symbols: Sequence[str],
) -> pa.Table:
    """Build a deterministic quote Arrow table from Yahoo v7 results.

    Columns: ``symbol`` then each requested field converted to snake_case
    (in caller order). Missing symbols get a null row; missing field values
    are null.
    """
    snake_fields: list[str]
    if fields is None:
        # Use whatever keys Yahoo returned (deduplicated, deterministic order).
        seen_keys: dict[str, None] = {}
        for row in quotes_data:
            for k in row:
                if k != "symbol":
                    seen_keys.setdefault(k, None)
        snake_fields = list(seen_keys)
    else:
        snake_fields = [camel_to_snake(f) for f in fields]

    # Build a lookup from Yahoo symbol -> result row
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in quotes_data:
        sym = row.get("symbol")
        if isinstance(sym, str):
            by_symbol[sym.upper()] = row

    symbols: list[str] = []
    column_values: dict[str, list[Any]] = {f: [] for f in snake_fields}

    for sym in requested_symbols:
        symbols.append(sym)
        row = by_symbol.get(sym)
        for idx, field_name in enumerate(snake_fields):
            camel = _snake_back_to_camel(fields[idx], field_name) if fields else field_name
            val = row.get(camel, None) if row else None
            column_values[field_name].append(val)

    col_arrays: list[pa.Array] = [pa.array(symbols, type=pa.string())]
    col_names = ["symbol"]
    for field_name in snake_fields:
        vals = column_values[field_name]
        arr = _infer_arrow_array(vals)
        col_arrays.append(arr)
        col_names.append(field_name)

    return pa.table(col_arrays, names=col_names)


def _snake_back_to_camel(camel_field: str, snake_field: str) -> str:
    """Given the original camelCase field and its snake form, return camel.

    The snake form is only used for column naming; for lookups we use the
    original camelCase name.
    """
    return camel_field


def _infer_arrow_array(values: list[Any]) -> pa.Array:
    """Best-effort inference of an Arrow array from a list of Python values.

    All-null arrays become string (safe default); numeric values use float64
    or int64; booleans use bool; everything else string.
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return pa.nulls(len(values), type=pa.string())

    first = non_null[0]
    all_bool = all(isinstance(v, bool) for v in non_null)
    if all_bool:
        return pa.array(values, type=pa.bool_())

    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in non_null)
    if all_int:
        return pa.array(values, type=pa.int64())

    all_float = all(isinstance(v, int | float) and not isinstance(v, bool) for v in non_null)
    if all_float:
        return pa.array(values, type=pa.float64())

    # Default: stringify
    return pa.array([str(v) if v is not None else None for v in values], type=pa.string())


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def build_history_table(
    symbol: str,
    chart_result: dict[str, Any] | None,
) -> pa.Table:
    """Convert a single Yahoo v8 chart ``result[0]`` payload into rows.

    Returns an empty table with ``HISTORY_SCHEMA`` when *chart_result* is None
    or contains no timestamps.
    """
    if chart_result is None:
        return _empty_history_table()

    meta = chart_result.get("meta", {})
    currency = meta.get("currency") or None
    exchange_tz = meta.get("exchangeTimezoneName") or meta.get("gmtOffSetMilliseconds") or None

    timestamps: list[int] | None = chart_result.get("timestamp")
    if not timestamps:
        return _empty_history_table()

    indicators = chart_result.get("indicators", {})

    # Primary quote (OHLCV) — may have multiple entries, use the first.
    quote_list = indicators.get("quote") or [{}]
    quote_data: dict[str, Any] = quote_list[0] if quote_list else {}

    opens = _pad(quote_data.get("open"), len(timestamps))
    highs = _pad(quote_data.get("high"), len(timestamps))
    lows = _pad(quote_data.get("low"), len(timestamps))
    closes = _pad(quote_data.get("close"), len(timestamps))
    volumes = _pad(quote_data.get("volume"), len(timestamps))

    # Adjusted close
    adjclose_list = indicators.get("adjclose") or []
    adjusted_closes: list[float | None]
    if adjclose_list:
        adjusted_closes = _pad(adjclose_list[0].get("adjclose"), len(timestamps))
    else:
        adjusted_closes = [None] * len(timestamps)

    # Events: dividends and splits
    events = chart_result.get("events", {})
    dividends_map = events.get("dividends", {}) or {}
    splits_map = events.get("splits", {}) or {}

    ts_set = set(timestamps)
    dividends_by_ts: dict[int, float] = {}
    for ev in dividends_map.values():
        ts = ev.get("date")
        amount = ev.get("amount")
        if ts is not None and amount is not None:
            dividends_by_ts[int(ts)] = float(amount)

    splits_by_ts: dict[int, float] = {}
    for ev in splits_map.values():
        ts = ev.get("date")
        num = ev.get("numerator")
        den = ev.get("denominator")
        ratio = ev.get("splitRatio")
        if ts is not None:
            if num is not None and den is not None:
                try:
                    splits_by_ts[int(ts)] = float(num) / float(den)
                except (TypeError, ZeroDivisionError):
                    pass
            elif ratio is not None and isinstance(ratio, str | int | float):
                try:
                    splits_by_ts[int(ts)] = _parse_split_ratio(ratio)
                except (ValueError, TypeError):
                    pass
    _ = ts_set  # keep for future validation

    dividend_col = [dividends_by_ts.get(ts) for ts in timestamps]
    split_col = [splits_by_ts.get(ts) for ts in timestamps]

    # Build arrays
    symbol_col = pa.array([symbol] * len(timestamps), type=pa.string())
    ts_col = pa.array(
        [dt.datetime.fromtimestamp(ts, tz=dt.UTC) for ts in timestamps],
        type=pa.timestamp("s", tz="UTC"),
    )
    open_col = _to_float_array(opens)
    high_col = _to_float_array(highs)
    low_col = _to_float_array(lows)
    close_col = _to_float_array(closes)
    adj_col = _to_float_array(adjusted_closes)
    vol_col = pa.array(volumes, type=pa.int64())
    div_col = _to_float_array(dividend_col)
    split_col_arr = _to_float_array(split_col)
    currency_col = pa.array([currency] * len(timestamps), type=pa.string())
    tz_col = pa.array([exchange_tz] * len(timestamps), type=pa.string())

    table = pa.table(
        {
            "symbol": symbol_col,
            "timestamp": ts_col,
            "open": open_col,
            "high": high_col,
            "low": low_col,
            "close": close_col,
            "adjusted_close": adj_col,
            "volume": vol_col,
            "dividend": div_col,
            "split_ratio": split_col_arr,
            "currency": currency_col,
            "exchange_timezone": tz_col,
        }
    )
    # Cast to the canonical schema for field order/types.
    return table.cast(HISTORY_SCHEMA)


def _parse_split_ratio(ratio: str | int | float) -> float:
    if isinstance(ratio, int | float):
        return float(ratio)
    # "2:1" form
    parts = ratio.split(":")
    if len(parts) == 2:
        return float(parts[0]) / float(parts[1])
    raise ValueError(f"Cannot parse split ratio: {ratio}")


def _pad(values: list[Any] | None, length: int) -> list[Any]:
    """Ensure *values* has exactly *length* elements, padding with None."""
    if values is None:
        return [None] * length
    if len(values) == length:
        return values
    if len(values) < length:
        return list(values) + [None] * (length - len(values))
    return list(values)[:length]


def _to_float_array(values: list[float | None]) -> pa.Array:
    return pa.array(values, type=pa.float64())


def _empty_history_table() -> pa.Table:
    """Return an empty history table with the canonical schema."""
    arrays = [pa.array([], type=field.type) for field in HISTORY_SCHEMA]
    return pa.table(arrays, schema=HISTORY_SCHEMA)


# ---------------------------------------------------------------------------
# Polars convenience
# ---------------------------------------------------------------------------

_POLARS_IMPORT_ERROR: str | None = None


def to_polars(table: pa.Table) -> Any:
    """Convert a pyarrow Table to a Polars DataFrame.

    Requires the ``polars`` optional dependency. Raises a clear
    :class:`ImportError` if Polars is not installed.
    """
    global _POLARS_IMPORT_ERROR
    if _POLARS_IMPORT_ERROR:
        raise ImportError(_POLARS_IMPORT_ERROR)
    try:
        import polars as pl
    except ImportError as exc:
        _POLARS_IMPORT_ERROR = (
            "Polars is not installed. Install it with: pip install 'yfin[polars]' or pip install polars"
        )
        raise ImportError(_POLARS_IMPORT_ERROR) from exc
    return pl.from_arrow(table)
