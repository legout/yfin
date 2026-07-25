"""Tests for the quoteSummary v10 provider.

Uses Yahoo v10 quoteSummary JSON fixtures. Covers: flat modules, multi-symbol,
multi-module merging, {"raw": X} extraction, failure skipping, array modules
(upgrade history), and the sync wrapper.
"""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from yfin.arrow import build_summary_table
from yfin.exceptions import YahooApiError
from yfin.models import YahooRoute

# ---------------------------------------------------------------------------
# Yahoo v10 quoteSummary fixtures
# ---------------------------------------------------------------------------

ASSET_PROFILE_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "assetProfile": {
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "fullTimeEmployees": 166000,
                    "longBusinessSummary": (
                        "Apple Inc. designs, manufactures and markets smartphones."
                    ),
                    "city": "Cupertino",
                    "state": "CA",
                    "country": "United States",
                    "website": "https://www.apple.com",
                    "address1": "One Apple Park Way",
                    "zip": "95014",
                    "phone": "(408) 996-1010",
                    "companyOfficers": [
                        {"name": "Tim Cook", "age": 64, "title": "CEO"},
                    ],
                    "auditRisk": 2,
                    "boardRisk": 1,
                    "compensationRisk": 7,
                    "overallRisk": 1,
                    "shareHolderRightsRisk": 1,
                    "governanceEpochDate": 1751308800,
                },
                "quoteType": {
                    "exchange": "NMS",
                    "longName": "Apple Inc.",
                    "shortName": "Apple Inc.",
                    "quoteType": "EQUITY",
                    "symbol": "AAPL",
                    "firstTradeDateEpochUtc": 345479400,
                    "timeZoneFullName": "America/New_York",
                    "timeZoneShortName": "EDT",
                    "uuid": "8b10e4ae-abcd-1234-5678-abcdef012345",
                    "gmtOffSetMilliseconds": -14400000,
                },
            }
        ],
        "error": None,
    }
}

SUMMARY_DETAIL_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "summaryDetail": {
                    "beta": 1.097,
                    "marketCap": {"raw": 4891182891008, "fmt": "4.89T"},
                    "trailingPE": {"raw": 40.31719, "fmt": "40.32"},
                    "forwardPE": {"raw": 34.541164, "fmt": "34.54"},
                    "dividendRate": {"raw": 1.08, "fmt": "1.08"},
                    "dividendYield": {"raw": 0.0032, "fmt": "0.32%"},
                    "fiftyTwoWeekHigh": {"raw": 334.99, "fmt": "334.99"},
                    "fiftyTwoWeekLow": {"raw": 201.5, "fmt": "201.50"},
                    "currency": "USD",
                    "volume": {"raw": 47402209, "fmt": "47.4M", "longFmt": "47,402,209"},
                    "tradeable": False,
                },
                "quoteType": {
                    "symbol": "AAPL",
                },
            }
        ],
        "error": None,
    }
}

DEFAULT_KEY_STATISTICS_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "defaultKeyStatistics": {
                    "beta": {"raw": 1.097, "fmt": "1.10"},
                    "bookValue": {"raw": 7.26, "fmt": "7.26"},
                    "forwardPE": {"raw": 34.541164, "fmt": "34.54"},
                    "forwardEps": {"raw": 9.64125, "fmt": "9.64"},
                    "enterpriseValue": {"raw": 4907387060224, "fmt": "4.91T"},
                    "floatShares": {"raw": 14662387495, "fmt": "14.66B"},
                    "heldPercentInsiders": {"raw": 0.0163, "fmt": "1.63%"},
                    "heldPercentInstitutions": {"raw": 0.66499, "fmt": "66.50%"},
                    "52WeekChange": {"raw": 0.5558047, "fmt": "55.58%"},
                    "dateShortInterest": "2026-07-15",
                    "lastSplitDate": None,
                    "lastSplitFactor": None,
                },
                "quoteType": {"symbol": "AAPL"},
            }
        ],
        "error": None,
    }
}

UPGRADE_DOWNGRADE_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "quoteType": {"symbol": "AAPL"},
                "upgradeDowngradeHistory": {
                    "history": [
                        {
                            "epochGradeDate": 1752624000,
                            "firm": "Morgan Stanley",
                            "toGrade": "Overweight",
                            "fromGrade": "Overweight",
                            "action": "main",
                        },
                        {
                            "epochGradeDate": 1752537600,
                            "firm": "HSBC",
                            "toGrade": "Buy",
                            "fromGrade": "Hold",
                            "action": "up",
                        },
                        {
                            "epochGradeDate": 1752451200,
                            "firm": "Barclays",
                            "toGrade": "Equal-Weight",
                            "fromGrade": "Overweight",
                            "action": "down",
                        },
                    ]
                },
            }
        ],
        "error": None,
    }
}

CALENDAR_EVENTS_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "quoteType": {"symbol": "AAPL"},
                "calendarEvents": {
                    "dividendDate": {"raw": 1747180800, "fmt": "2025-05-14"},
                    "exDividendDate": {"raw": 1746921600, "fmt": "2025-05-11"},
                    "earnings": {
                        "earningsDate": [
                            {"raw": 1753920000, "fmt": "2025-07-31"},
                            {"raw": 1754006400, "fmt": "2025-08-01"},
                        ],
                        "earningsAverage": {"raw": 1.85, "fmt": "1.85"},
                        "earningsLow": {"raw": 1.75, "fmt": "1.75"},
                        "earningsHigh": {"raw": 1.95, "fmt": "1.95"},
                        "revenueAverage": {"raw": 128000000000, "fmt": "128B"},
                    },
                },
            }
        ],
        "error": None,
    }
}

INSTITUTION_OWNERSHIP_RESPONSE = {
    "quoteSummary": {
        "result": [
            {
                "quoteType": {"symbol": "AAPL"},
                "institutionOwnership": {
                    "ownershipList": [
                        {
                            "organization": "Vanguard Group Inc",
                            "pctHeld": {"raw": 0.0812, "fmt": "8.12%"},
                            "position": {"raw": 1200000000, "fmt": "1.2B"},
                            "value": {"raw": 390000000000, "fmt": "390B"},
                            "reportDate": 1747180800,
                        },
                        {
                            "organization": "BlackRock Inc.",
                            "pctHeld": {"raw": 0.0658, "fmt": "6.58%"},
                            "position": {"raw": 970000000, "fmt": "970M"},
                            "value": {"raw": 315000000000, "fmt": "315B"},
                            "reportDate": 1747180800,
                        },
                    ]
                },
            }
        ],
        "error": None,
    }
}

ERROR_RESPONSE = {
    "quoteSummary": {
        "result": None,
        "error": {
            "code": "Not Found",
            "description": "No data found, symbol may be delisted or no longer traded",
        },
    }
}


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


class _MockSummaryClient:
    """Mock client that returns different responses per symbol."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses
        self.closed = False

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        # Extract symbol from URL: .../quoteSummary/AAPL
        symbol = url.rsplit("/", 1)[-1].upper()
        resp = self._responses.get(symbol)
        if resp is None:
            return ERROR_RESPONSE
        return resp

    async def close(self) -> None:
        self.closed = True


class _ErrorSummaryClient:
    """Mock that raises on get_json."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def get_route(self, proxy: str | None = None) -> YahooRoute:
        return YahooRoute(proxy=proxy or "")

    async def get_json(self, url: str, params: dict | None = None, route=None) -> dict:
        raise self._error

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# build_summary_table (Arrow builder unit tests)
# ---------------------------------------------------------------------------


class TestBuildSummaryTable:
    def test_asset_profile_flat(self) -> None:
        entry = ASSET_PROFILE_RESPONSE["quoteSummary"]["result"][0]
        table = build_summary_table([entry], ["assetProfile", "quoteType"])
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1
        assert table.column_names[0] == "symbol"
        assert "asset_profile_sector" in table.column_names
        assert "asset_profile_industry" in table.column_names
        assert "asset_profile_full_time_employees" in table.column_names

        sector = table.column("asset_profile_sector").to_pylist()
        assert sector == ["Technology"]
        industry = table.column("asset_profile_industry").to_pylist()
        assert industry == ["Consumer Electronics"]
        employees = table.column("asset_profile_full_time_employees").to_pylist()
        assert employees == [166000]

    def test_raw_extraction(self) -> None:
        """Values wrapped as {"raw": X} → extracted to X."""
        entry = SUMMARY_DETAIL_RESPONSE["quoteSummary"]["result"][0]
        table = build_summary_table([entry], ["summaryDetail"])
        assert table.num_rows == 1

        market_cap = table.column("summary_detail_market_cap").to_pylist()
        assert market_cap == [4891182891008]

        trailing_pe = table.column("summary_detail_trailing_pe").to_pylist()
        assert trailing_pe == [40.31719]

        volume = table.column("summary_detail_volume").to_pylist()
        assert volume == [47402209]

        # Plain scalar stays as-is
        currency = table.column("summary_detail_currency").to_pylist()
        assert currency == ["USD"]

        # Plain bool stays bool
        tradeable = table.column("summary_detail_tradeable").to_pylist()
        assert tradeable == [False]

    def test_multi_symbol(self) -> None:
        """2 symbols, same module."""
        aapl = {
            "quoteType": {"symbol": "AAPL"},
            "assetProfile": {"sector": "Technology", "industry": "Consumer Electronics"},
        }
        msft = {
            "quoteType": {"symbol": "MSFT"},
            "assetProfile": {"sector": "Technology", "industry": "Software"},
        }
        table = build_summary_table([aapl, msft], ["assetProfile"])
        assert table.num_rows == 2
        symbols = table.column("symbol").to_pylist()
        assert symbols == ["AAPL", "MSFT"]
        industries = table.column("asset_profile_industry").to_pylist()
        assert industries == ["Consumer Electronics", "Software"]

    def test_multi_module_merged(self) -> None:
        """1 symbol, 2 modules merged into one wide row."""
        entry = {
            "quoteType": {"symbol": "AAPL"},
            "summaryDetail": {"beta": 1.097, "marketCap": {"raw": 4000, "fmt": "4K"}},
            "defaultKeyStatistics": {"bookValue": {"raw": 7.26, "fmt": "7.26"}},
        }
        table = build_summary_table([entry], ["summaryDetail", "defaultKeyStatistics"])
        assert table.num_rows == 1
        assert "summary_detail_beta" in table.column_names
        assert "summary_detail_market_cap" in table.column_names
        assert "default_key_statistics_book_value" in table.column_names

    def test_upgrade_history_array(self) -> None:
        """Array module produces multiple rows (one per event)."""
        entry = UPGRADE_DOWNGRADE_RESPONSE["quoteSummary"]["result"][0]
        table = build_summary_table([entry], ["upgradeDowngradeHistory"])
        assert table.num_rows == 3
        symbols = table.column("symbol").to_pylist()
        assert all(s == "AAPL" for s in symbols)
        firms = table.column("upgrade_downgrade_history_firm").to_pylist()
        assert "Morgan Stanley" in firms
        assert "HSBC" in firms
        assert "Barclays" in firms
        to_grades = table.column("upgrade_downgrade_history_to_grade").to_pylist()
        assert to_grades == ["Overweight", "Buy", "Equal-Weight"]

    def test_institution_ownership_array(self) -> None:
        entry = INSTITUTION_OWNERSHIP_RESPONSE["quoteSummary"]["result"][0]
        table = build_summary_table([entry], ["institutionOwnership"])
        assert table.num_rows == 2
        orgs = table.column("institution_ownership_organization").to_pylist()
        assert orgs == ["Vanguard Group Inc", "BlackRock Inc."]
        pct = table.column("institution_ownership_pct_held").to_pylist()
        assert pct == [0.0812, 0.0658]

    def test_calendar_events(self) -> None:
        entry = CALENDAR_EVENTS_RESPONSE["quoteSummary"]["result"][0]
        table = build_summary_table([entry], ["calendarEvents"])
        assert table.num_rows == 1
        div_date = table.column("calendar_events_dividend_date").to_pylist()
        assert div_date == [1747180800]
        ex_div = table.column("calendar_events_ex_dividend_date").to_pylist()
        assert ex_div == [1746921600]
        # First earnings date extracted from list
        ed = table.column("calendar_events_earnings_date").to_pylist()
        assert ed == [1753920000]
        ea = table.column("calendar_events_earnings_earnings_average").to_pylist()
        assert ea == [1.85]

    def test_empty_raw_data(self) -> None:
        table = build_summary_table([], ["assetProfile"])
        assert table.num_rows == 0
        assert "symbol" in table.column_names


# ---------------------------------------------------------------------------
# quote_summary_async with mock client
# ---------------------------------------------------------------------------


class TestQuoteSummaryAsync:
    async def test_summary_asset_profile(self) -> None:
        from yfin.summary import asset_profile_async

        client = _MockSummaryClient({"AAPL": ASSET_PROFILE_RESPONSE})
        table = await asset_profile_async(["AAPL"], client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1
        assert table.column_names[0] == "symbol"
        assert "asset_profile_sector" in table.column_names
        assert table.column("symbol").to_pylist() == ["AAPL"]
        assert table.column("asset_profile_sector").to_pylist() == ["Technology"]

    async def test_summary_multi_symbol(self) -> None:
        from yfin.summary import key_statistics_async

        aapl_resp = DEFAULT_KEY_STATISTICS_RESPONSE
        msft_resp = {
            "quoteSummary": {
                "result": [
                    {
                        "defaultKeyStatistics": {
                            "beta": {"raw": 0.9, "fmt": "0.90"},
                            "bookValue": {"raw": 30.0, "fmt": "30.00"},
                        },
                        "quoteType": {"symbol": "MSFT"},
                    }
                ],
                "error": None,
            }
        }
        client = _MockSummaryClient({"AAPL": aapl_resp, "MSFT": msft_resp})
        table = await key_statistics_async(["AAPL", "MSFT"], client=client)
        assert table.num_rows == 2
        symbols = table.column("symbol").to_pylist()
        assert symbols == ["AAPL", "MSFT"]

    async def test_summary_multi_module(self) -> None:
        from yfin.summary import summary_detail_async

        # summary_detail_async requests summaryDetail + defaultKeyStatistics + financialData
        resp = {
            "quoteSummary": {
                "result": [
                    {
                        "quoteType": {"symbol": "AAPL"},
                        "summaryDetail": {"beta": 1.097, "fiftyTwoWeekHigh": {"raw": 334.99}},
                        "defaultKeyStatistics": {"bookValue": {"raw": 7.26}},
                        "financialData": {
                            "currentPrice": {"raw": 333.02, "fmt": "333.02"},
                            "targetHighPrice": {"raw": 400.0, "fmt": "400.00"},
                            "financialCurrency": "USD",
                        },
                    }
                ],
                "error": None,
            }
        }
        client = _MockSummaryClient({"AAPL": resp})
        table = await summary_detail_async(["AAPL"], client=client)
        assert table.num_rows == 1
        # Columns from all three flat modules present
        assert "summary_detail_beta" in table.column_names
        assert "default_key_statistics_book_value" in table.column_names
        assert "financial_data_current_price" in table.column_names
        assert "financial_data_financial_currency" in table.column_names

    async def test_summary_raw_extraction(self) -> None:
        from yfin.summary import quote_summary_async

        client = _MockSummaryClient({"AAPL": SUMMARY_DETAIL_RESPONSE})
        table = await quote_summary_async(["AAPL"], ["summaryDetail"], client=client)
        assert table.num_rows == 1
        # {"raw": 4891182891008} → 4891182891008
        mc = table.column("summary_detail_market_cap").to_pylist()
        assert mc == [4891182891008]
        # {"raw": 40.31719} → 40.31719
        pe = table.column("summary_detail_trailing_pe").to_pylist()
        assert pe == [40.31719]

    async def test_summary_skips_failed_symbol(self) -> None:
        from yfin.summary import quote_summary_async

        # AAPL succeeds, BAD raises via ERROR_RESPONSE being returned but that
        # is a valid payload shape — we need the client to raise instead.
        _MockSummaryClient(
            {"AAPL": ASSET_PROFILE_RESPONSE},
        )
        # Now patch: make MSFT raise by using an error-raising client for it.
        # We simulate one symbol failing by using an _ErrorSummaryClient for
        # one of two symbols via a composite client.

        class _CompositeClient:
            def __init__(self) -> None:
                self.good = _MockSummaryClient({"AAPL": ASSET_PROFILE_RESPONSE})
                self.bad = _ErrorSummaryClient(YahooApiError("boom"))

            def get_route(self, proxy: str | None = None) -> YahooRoute:
                return YahooRoute(proxy=proxy or "")

            async def get_json(self, url, params=None, route=None):
                symbol = url.rsplit("/", 1)[-1].upper()
                if symbol == "AAPL":
                    return await self.good.get_json(url, params, route)
                return await self.bad.get_json(url, params, route)

            async def close(self) -> None:
                pass

        client = _CompositeClient()
        table = await quote_summary_async(["AAPL", "MSFT"], ["assetProfile"], client=client)
        # MSFT failed and was skipped; only AAPL row present
        assert table.num_rows == 1
        assert table.column("symbol").to_pylist() == ["AAPL"]

    async def test_summary_upgrade_history(self) -> None:
        from yfin.summary import upgrade_downgrade_history_async

        client = _MockSummaryClient({"AAPL": UPGRADE_DOWNGRADE_RESPONSE})
        table = await upgrade_downgrade_history_async(["AAPL"], client=client)
        assert table.num_rows == 3
        firms = table.column("upgrade_downgrade_history_firm").to_pylist()
        assert "Morgan Stanley" in firms
        assert "HSBC" in firms
        assert "Barclays" in firms
        assert all(s == "AAPL" for s in table.column("symbol").to_pylist())

    async def test_summary_single_module_string(self) -> None:
        """A single module passed as a string is accepted."""
        from yfin.summary import quote_summary_async

        client = _MockSummaryClient({"AAPL": ASSET_PROFILE_RESPONSE})
        table = await quote_summary_async(["AAPL"], "assetProfile", client=client)
        assert table.num_rows == 1
        assert "asset_profile_sector" in table.column_names

    async def test_summary_empty_modules_raises(self) -> None:
        from yfin.summary import quote_summary_async

        with pytest.raises(ValueError, match="At least one module"):
            await quote_summary_async(["AAPL"], [], client=_MockSummaryClient({}))

    async def test_summary_all_symbols_fail_raises(self) -> None:
        from yfin.summary import quote_summary_async

        client = _ErrorSummaryClient(YahooApiError("network failure"))
        with pytest.raises(YahooApiError, match="All 1 symbols failed"):
            await quote_summary_async(["AAPL"], ["assetProfile"], client=client)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class TestSummarySync:
    def test_summary_sync_wrapper(self) -> None:
        from yfin.summary import asset_profile

        client = _MockSummaryClient({"AAPL": ASSET_PROFILE_RESPONSE})
        table = asset_profile(["AAPL"], client=client)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1
        assert table.column("symbol").to_pylist() == ["AAPL"]
        assert table.column("asset_profile_sector").to_pylist() == ["Technology"]

    def test_summary_generic_sync_wrapper(self) -> None:
        from yfin.summary import quote_summary

        client = _MockSummaryClient({"AAPL": SUMMARY_DETAIL_RESPONSE})
        table = quote_summary(["AAPL"], ["summaryDetail"], client=client)
        assert table.num_rows == 1
        mc = table.column("summary_detail_market_cap").to_pylist()
        assert mc == [4891182891008]

    def test_summary_sync_fails_inside_running_loop(self) -> None:
        from yfin.summary import asset_profile

        async def _run_in_loop() -> None:
            with pytest.raises(RuntimeError, match="running event loop"):
                asset_profile(["AAPL"], client=_MockSummaryClient({"AAPL": ASSET_PROFILE_RESPONSE}))

        asyncio.run(_run_in_loop())
