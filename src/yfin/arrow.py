"""Arrow table construction for yfin.

These functions convert validated Yahoo JSON payloads into deterministic
``pyarrow.Table`` objects whose schemas are documented in :mod:`yfin.models`.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from .models import HISTORY_SCHEMA, camel_to_snake

__all__ = [
    "build_quote_table",
    "build_history_table",
    "build_summary_table",
    "build_fundamentals_table",
    "INTEGER_TYPES",
    "to_polars",
]

# Types that Yahoo returns as integer-valued. All other fundamental types
# are float-valued. Defined here (rather than in fundamentals.py) because
# build_fundamentals_table needs it and arrow.py sits at the bottom of the
# import chain.
INTEGER_TYPES: frozenset[str] = frozenset(
    {"MarketCap", "EnterpriseValue", "BasicAverageShares", "DilutedAverageShares"}
)


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
    # Build (camel_for_lookup, snake_for_column) pairs
    field_pairs: list[tuple[str, str]]
    if fields is None:
        # Use whatever keys Yahoo returned (deduplicated, deterministic order).
        seen_keys: dict[str, None] = {}
        for row in quotes_data:
            for k in row:
                if k != "symbol":
                    seen_keys.setdefault(k, None)
        field_pairs = [(k, camel_to_snake(k)) for k in seen_keys]
    else:
        field_pairs = [(f, camel_to_snake(f)) for f in fields]

    snake_fields = [pair[1] for pair in field_pairs]

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
        for camel_key, snake_name in field_pairs:
            val = row.get(camel_key, None) if row else None
            column_values[snake_name].append(val)

    col_arrays: list[pa.Array] = [pa.array(symbols, type=pa.string())]
    col_names = ["symbol"]
    for field_name in snake_fields:
        vals = column_values[field_name]
        arr = _infer_arrow_array(vals)
        col_arrays.append(arr)
        col_names.append(field_name)

    return pa.table(col_arrays, names=col_names)


def _infer_arrow_array(values: list[Any]) -> pa.Array:
    """Best-effort inference of an Arrow array from a list of Python values.

    All-null arrays become string (safe default); numeric values use float64
    or int64; booleans use bool; everything else string.
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return pa.nulls(len(values), type=pa.string())

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
                with contextlib.suppress(TypeError, ZeroDivisionError):
                    splits_by_ts[int(ts)] = float(num) / float(den)
            elif ratio is not None and isinstance(ratio, str | int | float):
                with contextlib.suppress(ValueError, TypeError):
                    splits_by_ts[int(ts)] = _parse_split_ratio(ratio)

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
# quoteSummary (v10)
# ---------------------------------------------------------------------------

# Modules whose payloads are a single flat dict of scalar fields.
_FLAT_SUMMARY_MODULES = frozenset(
    {
        "assetProfile",
        "quoteType",
        "summaryDetail",
        "defaultKeyStatistics",
        "financialData",
    }
)

# Modules whose payloads contain a list of per-event dicts. The value is the
# key inside the module dict that holds the event list.
_ARRAY_SUMMARY_MODULES: dict[str, str] = {
    "upgradeDowngradeHistory": "history",
    "institutionOwnership": "ownershipList",
    "fundOwnership": "ownershipList",
    "insiderTransactions": "transactions",
    "insiderHolders": "holders",
    "majorHoldersBreakdown": "holders",
    "recommendationTrend": "trend",
}

# Calendar module is flat-with-nested-earnings; handled specially below.
_CALENDAR_MODULE = "calendarEvents"


def _extract_raw(value: Any) -> Any:
    """Unwrap a Yahoo ``{"raw": X, "fmt": ...}`` wrapper to *X*.

    Returns the value unchanged when it is not a raw-wrapper. Recurses into
    lists so that arrays of raw-wrapped values are flattened element-wise.
    """
    if isinstance(value, dict):
        if "raw" in value:
            return value["raw"]
        return value
    if isinstance(value, list):
        return [_extract_raw(v) for v in value]
    return value


def build_summary_table(
    raw_data: list[dict[str, Any]],
    modules: list[str],
) -> pa.Table:
    """Build a deterministic Arrow table from quoteSummary v10 results.

    Each entry in *raw_data* is one symbol's ``quoteSummary.result[0]`` dict
    (a mapping of module name → module payload). *modules* is the list of
    module names that were requested and should be materialised as columns.

    FLAT modules (assetProfile, summaryDetail, …) and CALENDAR modules produce
    one combined row per symbol — all scalar fields are merged into a single
    wide row.

    ARRAY modules (upgradeDowngradeHistory, institutionOwnership, …) produce
    one row per event, each tagged with the symbol.

    The per-symbol flat table and the per-event array table are concatenated
    row-wise (array rows carry nulls for flat columns and vice versa).
    """
    if not raw_data:
        return _empty_summary_table()

    scalar_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []

    for entry in raw_data:
        symbol = _entry_symbol(entry)
        # Merge all flat + calendar modules for this symbol into one row.
        scalar_row: dict[str, Any] = {"symbol": symbol}
        has_scalar = False
        for module in modules:
            payload = entry.get(module)
            if not isinstance(payload, dict):
                continue
            if module in _FLAT_SUMMARY_MODULES:
                scalar_row.update(_flatten_module(symbol, module, payload))
                has_scalar = True
            elif module == _CALENDAR_MODULE:
                scalar_row.update(_flatten_calendar(symbol, payload))
                has_scalar = True
            elif module in _ARRAY_SUMMARY_MODULES:
                list_key = _ARRAY_SUMMARY_MODULES[module]
                events = payload.get(list_key)
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict):
                            array_rows.append(_flatten_event(symbol, module, list_key, event))
        if has_scalar:
            scalar_rows.append(scalar_row)

    tables: list[pa.Table] = []
    if scalar_rows:
        tables.append(_rows_to_table(scalar_rows))
    if array_rows:
        tables.append(_rows_to_table(array_rows))

    if not tables:
        return _empty_summary_table()
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def _entry_symbol(entry: dict[str, Any]) -> str:
    """Best-effort extraction of the symbol from a result entry."""
    for key in ("quoteType", "price", "summaryDetail"):
        sub = entry.get(key)
        if isinstance(sub, dict):
            sym = sub.get("symbol")
            if isinstance(sym, str):
                return sym.upper()
    return ""


def _flatten_module(
    symbol: str,
    module: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Flatten a flat module dict into a single row.

    Nested ``{"raw": X}`` values are unwrapped. Complex values (lists/dicts)
    are dropped — they are handled by the array/calendar paths. Keys are
    prefixed with the module name to avoid collisions across modules.
    """
    row: dict[str, Any] = {"symbol": symbol}
    for key, value in payload.items():
        if key == "symbol":
            continue
        unwrapped = _extract_raw(value)
        if isinstance(unwrapped, (dict, list)):
            # companyOfficers etc. — skip in the flat view.
            continue
        col = f"{module}.{key}"
        row[col] = unwrapped
    return row


def _flatten_event(
    symbol: str,
    module: str,
    list_key: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Flatten a single array-module event into one row."""
    row: dict[str, Any] = {"symbol": symbol}
    for key, value in event.items():
        unwrapped = _extract_raw(value)
        if isinstance(unwrapped, dict):
            # Nested objects inside events: flatten one level.
            for sub_k, sub_v in unwrapped.items():
                row[f"{module}.{key}.{sub_k}"] = _extract_raw(sub_v)
        else:
            row[f"{module}.{key}"] = unwrapped
    return row


def _flatten_calendar(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten the calendarEvents module into one row per symbol."""
    row: dict[str, Any] = {"symbol": symbol}
    for key, value in payload.items():
        if key == "earnings" and isinstance(value, dict):
            earnings = value
            # earningsDate is a list of {"raw": epoch} objects.
            ed = earnings.get("earningsDate")
            if isinstance(ed, list) and ed:
                first = ed[0] if isinstance(ed[0], dict) else {}
                row["calendarEvents.earningsDate"] = _extract_raw(first)
            for ek, ev in earnings.items():
                if ek == "earningsDate":
                    continue
                row[f"calendarEvents.earnings.{ek}"] = _extract_raw(ev)
        else:
            row[f"calendarEvents.{key}"] = _extract_raw(value)
    return row


def _rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build an Arrow table from a list of row dicts.

    Column order is deterministic: ``symbol`` first, then remaining keys in
    first-seen order. Missing values in later rows become null.
    """
    ordered_fields: list[str] = ["symbol"]
    seen: set[str] = {"symbol"}
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                ordered_fields.append(k)

    col_arrays: list[pa.Array] = []
    col_names: list[str] = []
    for field in ordered_fields:
        values = [row.get(field) for row in rows]
        if field == "symbol":
            col_arrays.append(pa.array(values, type=pa.string()))
        else:
            col_arrays.append(_infer_arrow_array(values))
        col_names.append(camel_to_snake(field.replace(".", "_")))

    return pa.table(col_arrays, names=col_names)


def _empty_summary_table() -> pa.Table:
    """Return an empty summary table with just a symbol column."""
    return pa.table({"symbol": pa.array([], type=pa.string())})


# ---------------------------------------------------------------------------
# Fundamentals timeseries
# ---------------------------------------------------------------------------


def build_fundamentals_table(
    raw_data: list[dict[str, Any]],
    types: list[str],
) -> pa.Table:
    """Build a deterministic Arrow table from fundamentals-timeseries rows.

    Parameters
    ----------
    raw_data
        Flat list of ``{symbol, asOfDate, <type1>: value, <type2>: value, ...}``
        dicts — one per (symbol, date) pair. Missing types are omitted from a
        row dict and become nulls in the output.
    types
        The Yahoo type names (camelCase) that were requested, in the order
        they should appear as columns.

    Returns
    -------
    pa.Table
        Columns: ``symbol`` (string), ``as_of_date`` (date32), then one column
        per type converted to snake_case. Integer types
        (:data:`INTEGER_TYPES`) use ``int64``; everything else ``float64``.
        An empty input yields an empty table with the full schema.
    """
    snake_types = [camel_to_snake(t) for t in types]
    is_int = [t in INTEGER_TYPES for t in types]

    # Column builders
    symbols: list[str | None] = []
    dates: list[dt.date | None] = []
    col_values: dict[str, list[Any]] = {t: [] for t in types}

    for row in raw_data:
        symbols.append(row.get("symbol"))
        as_of = row.get("asOfDate")
        dates.append(_parse_as_of_date(as_of) if as_of is not None else None)
        for t in types:
            col_values[t].append(row.get(t))

    col_arrays: list[pa.Array] = [pa.array(symbols, type=pa.string())]
    col_names: list[str] = ["symbol"]
    col_arrays.append(
        pa.array(dates, type=pa.date32()) if dates else pa.array([], type=pa.date32())
    )
    col_names.append("as_of_date")

    for t, snake_name, int_col in zip(types, snake_types, is_int, strict=True):
        vals = col_values[t]
        if int_col:
            col_arrays.append(pa.array(vals, type=pa.int64()))
        else:
            col_arrays.append(pa.array(vals, type=pa.float64()))
        col_names.append(snake_name)

    return pa.table(col_arrays, names=col_names)


def _parse_as_of_date(value: Any) -> dt.date | None:
    """Parse a Yahoo ``asOfDate`` (``"YYYY-MM-DD"``) into ``datetime.date``."""
    if isinstance(value, dt.date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


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
            "Polars is not installed. "
            "Install it with: pip install 'yfin[polars]' or pip install polars"
        )
        raise ImportError(_POLARS_IMPORT_ERROR) from exc
    return pl.from_arrow(table)
