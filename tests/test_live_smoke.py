"""Opt-in live Yahoo Finance smoke test.

This test is NEVER run automatically. It requires the environment variable
``YFIN_LIVE_SMOKE=1`` to be set, and performs real network requests to Yahoo.

Run manually:

    YFIN_LIVE_SMOKE=1 uv run pytest tests/test_live_smoke.py -q -s

The hermetic test suite (tests/test_auth.py, test_models.py, test_quotes.py,
test_history.py, test_client.py) must always pass without this test.
"""

from __future__ import annotations

import os
from datetime import date

import pyarrow as pa
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("YFIN_LIVE_SMOKE") != "1",
    reason="Set YFIN_LIVE_SMOKE=1 to run live Yahoo smoke tests",
)

_LIVE_SYMBOLS = ["AAPL", "MSFT"]


class TestLiveQuotes:
    async def test_live_quotes(self) -> None:
        import yfin

        table = await yfin.quotes_async(
            _LIVE_SYMBOLS,
            fields=["regularMarketPrice", "regularMarketVolume", "currency"],
        )
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert table.column_names[0] == "symbol"
        symbols = table.column("symbol").to_pylist()
        assert "AAPL" in symbols

    async def test_live_quotes_default_fields(self) -> None:
        import yfin

        table = await yfin.quotes_async(["AAPL"])
        assert isinstance(table, pa.Table)
        assert table.num_rows >= 1


class TestLiveHistory:
    async def test_live_history(self) -> None:
        import yfin

        table = await yfin.history_async(
            _LIVE_SYMBOLS,
            start=date(2024, 1, 1),
            end=date(2024, 6, 1),
        )
        assert isinstance(table, pa.Table)
        assert table.num_rows > 0
        assert table.schema == yfin.HISTORY_SCHEMA
        symbols = set(table.column("symbol").to_pylist())
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    async def test_live_history_range(self) -> None:
        import yfin

        table = await yfin.history_async(["AAPL"], period="1mo")
        assert isinstance(table, pa.Table)
        assert table.num_rows > 0
