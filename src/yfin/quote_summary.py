import asyncio
import datetime as dt
import math

import numpy as np
import pandas as pd
from parallel_requests import parallel_requests_async

from .constants import ALL_MODULES, URLS
from .utils.base import camel_to_snake, snake_to_camel


def _convert_date(date: int | list) -> dt.datetime | list:
    if isinstance(date, list):
        if isinstance(date[0], (int, float)):
            date = [
                dt.datetime.utcfromtimestamp(int(_date)) if _date else _date
                for _date in date
            ]
        else:
            [
                dt.datetime.utcfromtimestamp(int(_date)) if _date else _date
                for _date in date
            ]

    elif isinstance(date, (int, float)):
        if not math.isnan(date):
            date = dt.datetime.utcfromtimestamp(int(date))

    return date


class QuoteSummary:
    _BASE_URL = URLS["quoteSummary"]

    def __init__(
        self,
        symbols: str | tuple | list,
        modules: str | tuple | list = [],
    ):
        if isinstance(symbols, str):
            symbols = [symbols]
        self.symbols = symbols
        if isinstance(modules, str):
            modules = [modules]

        _modules = [
            module for module in snake_to_camel(modules) if module in ALL_MODULES.keys()
        ]
        self._modules = camel_to_snake(_modules)
        self._symbols = symbols  # dict.fromkeys(self._modules, [])
        self.results = dict()

    async def fetch(
        self, modules: str | list | None = None, parse: bool = True, *args, **kwargs
    ):
        def _parse(response: object) -> dict:
            if response["quoteSummary"]["result"] is None:
                result = None
            else:
                res = response["quoteSummary"]["result"][0]
                # modules = list(res.keys())
                for module in self._modules:
                    if snake_to_camel(module) not in res:
                        res[module] = None
                    else:
                        if module != snake_to_camel(module):
                            res[module] = res[snake_to_camel(module)]
                            res.pop(snake_to_camel(module))

                result = res

            return result

        self._url = [self._BASE_URL + symbols for symbols in self.symbols]

        if modules is not None:
            if isinstance(modules, str):
                modules = [modules]

            _modules = [
                module
                for module in snake_to_camel(modules)
                if module in ALL_MODULES.keys()
            ]
            self._modules.extend(camel_to_snake(_modules))
            # self._symbols.update(dict.fromkeys(_modules, []))

        if self._modules is None:
            self._modules = camel_to_snake(list(ALL_MODULES.keys()))

        self._modules = sorted(set(self._modules))

        if hasattr(self, "_results_raw"):
            _modules = [
                module
                for module in self._modules
                if not self._module_already_fetched(module)
            ]

        else:
            _modules = self._modules

        if len(_modules) > 0:
            _modules = ",".join(snake_to_camel(_modules))

            params = {
                "modules": _modules,
                "formattresult": "false",
                "lang": "en-US",
                "region": "US",
                "corsDomain": "finance.yahoo.com",
            }

            results = await parallel_requests_async(
                urls=self._url,
                params=params,
                keys=self.symbols,
                parse_func=_parse if parse else None,
                *args,
                **kwargs,
            )

            # remove empy results
            self._removed_symbols = [
                symbols for symbols in results if results[symbols] is None
            ]
            self._symbols = [
                symbol for symbol in results if results[symbol] is not None
            ]
            results = {symbol: results[symbol] for symbol in self._symbols}

            if hasattr(self, "_results_raw"):
                for symbol in results:
                    if symbol in self._results_raw:
                        self._results_raw[symbol].update(results[symbol])
                    else:
                        self._results_raw[symbol] = results[symbol]
            else:
                self._results_raw = results
        else:
            print("Nothing todo. Module already fetched or module unknown.")

    def _module_already_fetched(self, module: str) -> bool:
        if not hasattr(self, "_results_raw"):
            return False
        else:
            return module in self._results_raw[self._symbols[0]].keys()

    def _module_already_formated(self, module: str) -> bool:
        if not hasattr(self, "results"):
            return False
        else:
            return module in self.results

    @staticmethod
    def _extract_raw(results: dict):
        if results is not None:
            if "maxAge" in results:
                results.pop("maxAge", None)

            results_ = dict()
            for k in results:
                if isinstance(results[k], dict):
                    if "raw" in results[k]:
                        results_[k] = results[k]["raw"]
                    elif results[k] == dict():
                        results_[k] = None
                    else:
                        results_[k] = results[k]
                elif isinstance(results[k], list):
                    if len(results[k]) > 0:
                        results[k] = results[k][0]
                        if "raw" in results[k]:
                            results_[k] = results[k]["raw"]
                        elif results[k] == dict():
                            results_[k] = np.nan
                        else:
                            results_[k] = results[k]
                    else:
                        results_[k] = results[k]
                else:
                    results_[k] = results[k]

            return results_

        return results

    @staticmethod
    def _to_dataframe(results: list, convert_dates: list = None) -> pd.DataFrame:
        results = pd.DataFrame(results)
        if convert_dates is not None:
            for col in convert_dates:
                if col in results:
                    results[col] = results[col].dropna().apply(_convert_date)
        for col in results.columns:
            if col not in convert_dates:
                results[col] = results[col].apply(pd.to_numeric, errors="ignore")
            # results = results.astype(dict.fromkeys(convert_dates, "datetime64[s]"))

        results.columns = camel_to_snake(results.columns.tolist())

        return results

    @staticmethod
    def _to_series(results: list, convert_dates: list = None) -> pd.DataFrame:
        results = pd.Series(results)
        if convert_dates is not None:
            for idx in convert_dates:
                if idx in results:
                    if results[idx]:
                        results[idx] = _convert_date(results[idx])

        results.index = camel_to_snake(results.index.tolist())

        return results

    def __format_earnings_trends(self, result: dict):
        formatted_results = dict()

        formatted_results["period"] = result["period"]
        formatted_results["date"] = result["endDate"]

        earnings_estimate = self._extract_raw(result["earningsEstimate"])
        earnings_estimate_keys = [
            "earnings_estimate_" + idx for idx in earnings_estimate.keys()
        ]
        earnings_estimate = dict(
            zip(earnings_estimate_keys, earnings_estimate.values())
        )
        formatted_results.update(earnings_estimate)

        revenue_estimate = self._extract_raw(result["revenueEstimate"])
        revenue_estimate_keys = [
            "revenue_estimate_" + idx for idx in revenue_estimate.keys()
        ]
        revenue_estimate = dict(zip(revenue_estimate_keys, revenue_estimate.values()))
        formatted_results.update(revenue_estimate)

        eps_revisions = self._extract_raw(result["epsRevisions"])
        eps_revisions_keys = ["eps_revisions_" + idx for idx in eps_revisions.keys()]
        eps_revisions = dict(zip(eps_revisions_keys, eps_revisions.values()))
        formatted_results.update(eps_revisions)

        eps_trend = self._extract_raw(result["epsTrend"])
        eps_trend_keys = ["eps_trend_" + idx for idx in eps_trend.keys()]
        eps_trend = dict(zip(eps_trend_keys, eps_trend.values()))
        formatted_results.update(eps_trend)

        return formatted_results

    def _format_earnings_trends(self):
        key = ALL_MODULES["earningsTrend"]["key"]
        convert_dates = ALL_MODULES["earningsTrend"]["convert_dates"]

        results = dict()
        for symbol in self._results_raw:
            if self._results_raw[symbol]["earnings_trend"] is not None:
                result = []
                for res in self._results_raw[symbol]["earnings_trend"][key]:
                    result.append(self.__format_earnings_trends(res))
                results[symbol] = self._to_dataframe(
                    result, convert_dates=convert_dates
                )

        self.results["earnings_trend"] = results

    def _format_earning(self):
        results = dict()
        for symbol in self._results_raw:
            if self._results_raw[symbol]["earnings"] is not None:
                results[symbol] = dict()

                results[symbol] = pd.DataFrame(
                    [
                        self._extract_raw(r)
                        for r in self._results_raw[symbol]["earnings"][
                            "financialsChart"
                        ]["yearly"]
                    ]
                )
                results[symbol] = pd.concat(
                    [
                        results[symbol],
                        pd.DataFrame(
                            [
                                self._extract_raw(r)
                                for r in self._results_raw[symbol]["earnings"][
                                    "financialsChart"
                                ]["quarterly"]
                            ]
                        ),
                    ]
                )

        self.results["earnings"] = results

    def _format_results(self, module: str):
        module_ = snake_to_camel(module)
        key = ALL_MODULES[module_]["key"]
        convert_dates = ALL_MODULES[module_]["convert_dates"]
        convert = ALL_MODULES[module_]["convert"]

        # if module == "earnings_trend":
        #    self._format_earnings_trends()

        # if module == "earnings":
        #    self._format_earning()

        if convert == "df":
            results = dict()
            for symbol in self._results_raw:
                if module in self._results_raw[symbol]:
                    if self._results_raw[symbol][module] is not None:
                        result = []
                        for rr in (
                            self._results_raw[symbol][module][key]
                            if key is not None
                            else self._results_raw[symbol][module]
                        ):
                            result.append(self._extract_raw(rr))

                        results[symbol] = self._to_dataframe(
                            result, convert_dates=convert_dates
                        )

        else:
            results = dict()
            for symbol in self._results_raw:
                if module in self._results_raw[symbol]:
                    if self._results_raw[symbol][module] is not None:
                        results[symbol] = self._to_series(
                            (
                                self._extract_raw(
                                    self._results_raw[symbol][module][key]
                                )
                                if key is not None
                                else self._extract_raw(
                                    self._results_raw[symbol][module]
                                )
                            ),
                            convert_dates=convert_dates,
                        )

        self.results[module] = results

    async def asset_profile_async(self, **kwargs):
        module = "asset_profile"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def balance_sheet_history_async(
        self, quarterly: bool = False, **kwargs
    ) -> pd.DataFrame:
        module = "balance_sheet_history"
        if quarterly:
            module += "_quarterly"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def calendar_events_async(self, **kwargs) -> pd.DataFrame:
        module = "calendar_events"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def cashflow_statement_history_async(
        self, quarterly: bool = False, **kwargs
    ) -> pd.DataFrame:
        module = "cashflow_statement_history"
        if quarterly:
            module += "_quarterly"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def default_key_statistics_async(self, **kwargs) -> pd.DataFrame:
        module = "default_key_statistics"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def earnings_async(self, **kwargs) -> pd.DataFrame:
        module = "earnings"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_earnings()

        return self.results[module]

    async def earnings_history_async(self, **kwargs) -> pd.DataFrame:
        module = "earnings_history"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def earnings_trend_async(self, **kwargs) -> pd.DataFrame:
        module = "earnings_trend"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_earnings_trends()

        return self.results[module]

    async def financial_data_async(self, **kwargs) -> pd.DataFrame:
        module = "financial_data"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def fund_ownership_async(self, **kwargs) -> pd.DataFrame:
        module = "fund_ownership"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def income_statement_history_async(
        self, quarterly: bool = False, **kwargs
    ) -> pd.DataFrame:
        module = "income_statement_history"
        if quarterly:
            module += "_quarterly"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def insider_transactions_async(self, **kwargs) -> pd.DataFrame:
        module = "insider_transactions"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def insider_holders_async(self, **kwargs) -> pd.DataFrame:
        module = "insider_holders"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def institution_ownership_async(self, **kwargs) -> pd.DataFrame:
        module = "institution_ownership"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def major_holders_breakdown_async(self, **kwargs) -> pd.DataFrame:
        module = "major_holders_breakdown"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def net_share_purchase_activity_async(self, **kwargs) -> pd.DataFrame:
        module = "net_share_purchase_activity"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def price_async(self, **kwargs) -> pd.DataFrame:
        module = "price"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def quote_type_async(self, **kwargs) -> pd.DataFrame:
        module = "quote_type"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def recommendation_trend_async(self, **kwargs) -> pd.DataFrame:
        module = "recommendation_trend"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def summary_detail_async(self, **kwargs) -> pd.DataFrame:
        module = "summary_detail"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    async def summary_profile_async(self, **kwargs) -> pd.DataFrame:
        module = "summary_profile"
        if not self._module_already_fetched(module):
            await self.fetch(modules=module, **kwargs)

        if not self._module_already_formated(module):
            self._format_results(module=module)

        return self.results[module]

    @property
    def asset_profile(self, **kwargs):
        return asyncio.run(self.asset_profile_async(**kwargs))

    @property
    def balance_sheet_history(self, **kwargs):
        return asyncio.run(self.balance_sheet_history_async(quarterly=False, **kwargs))

    @property
    def balance_sheet_history_quarterly(self, **kwargs):
        return asyncio.run(self.balance_sheet_history_async(quarterly=True, **kwargs))

    @property
    def calendar_events(self, **kwargs):
        return asyncio.run(self.calendar_events_async(**kwargs))

    @property
    def cashflow_statement_history(self, **kwargs):
        return asyncio.run(
            self.cashflow_statement_history_async(quarterly=False, **kwargs)
        )

    @property
    def cashflow_statement_history_quarterly(self, **kwargs):
        return asyncio.run(
            self.cashflow_statement_history_async(quarterly=True, **kwargs)
        )

    @property
    def default_key_statistics(self, **kwargs):
        return asyncio.run(self.default_key_statistics_async(**kwargs))

    @property
    def earnings(self, **kwargs):
        return asyncio.run(self.earnings_async(**kwargs))

    @property
    def earnings_history(self, **kwargs):
        return asyncio.run(self.earnings_history_async(**kwargs))

    @property
    def earnings_trend(self, **kwargs):
        return asyncio.run(self.earnings_trend_async(**kwargs))

    @property
    def financial_data(self, **kwargs):
        return asyncio.run(self.financial_data_async(**kwargs))

    @property
    def fund_ownership(self, **kwargs):
        return asyncio.run(self.fund_ownership_async(**kwargs))

    @property
    def income_statement_history(self, **kwargs):
        return asyncio.run(
            self.income_statement_history_async(quarterly=False, **kwargs)
        )

    @property
    def income_statement_history_quarterly(self, **kwargs):
        return asyncio.run(
            self.income_statement_history_async(quarterly=True, **kwargs)
        )

    @property
    def insider_holders(self, **kwargs):
        return asyncio.run(self.insider_holders_async(**kwargs))

    @property
    def insider_transactions(self, **kwargs):
        return asyncio.run(self.insider_transactions_async(**kwargs))

    @property
    def institution_ownership(self, **kwargs):
        return asyncio.run(self.institution_ownership_async(**kwargs))

    @property
    def major_holders_breakdown(self, **kwargs):
        return asyncio.run(self.major_holders_breakdown_async(**kwargs))

    @property
    def net_share_purchase_activity(self, **kwargs):
        return asyncio.run(self.net_share_purchase_activity_async(**kwargs))

    @property
    def price(self, **kwargs):
        return asyncio.run(self.price_async(**kwargs))

    @property
    def quote_type(self, **kwargs):
        return asyncio.run(self.quote_type_async(**kwargs))

    @property
    def recommendation_trend(self, **kwargs):
        return asyncio.run(self.recommendation_trend_async(**kwargs))

    @property
    def summary_detail(self, **kwargs):
        return asyncio.run(self.summary_detail_async(**kwargs))

    @property
    def summary_profile(self, **kwargs):
        return asyncio.run(self.summary_profile_async(**kwargs))


async def quote_summary_async(
    symbols: str | list[str], modules: str | list[str] | None = None, *args, **kwargs
):
    qs = QuoteSummary(symbols=symbols, modules=modules)
    await qs.fetch(*args, **kwargs)
    results = {}
    if "earnings" in qs._modules:
        qs._format_earning()
        qs._modules.remove("earnings")
        results["earnings"] = (
            pd.concat(qs.results["earnings"], names=["symbol"])
            .reset_index()
            .drop("level_1", axis=1)
        )

    if "earnings_trend" in qs._modules:
        qs._format_earnings_trends()
        qs._modules.remove("earnings_trend")
        results["earnings_trend"] = (
            pd.concat(qs.results["earnings_trend"], names=["symbol"])
            .reset_index()
            .drop("level_1", axis=1)
        )

    for module in qs._modules:
        qs._format_results(module=module)
        if len(qs.results[module]):
            if isinstance(qs.results[module][list(qs.results[module].keys())[0]], pd.Series):
                res = pd.concat(qs.results[module])
                if "symbol" not in res.index.levels[1]:
                    res.index.names = ["symbol", None]
                    res = res.unstack().reset_index()
                else:
                    res = res.unstack().reset_index(drop=True)
                    
            elif isinstance(qs.results[module][list(qs.results[module].keys())[0]], pd.DataFrame):
                res = pd.concat(qs.results[module], names=["symbol"])
                if "symbol" not in res.columns:
                    res = res.reset_index().drop("level_1", axis=1)
                else:
                    res = res.reset_index(drop=True)
                    
            else:
                res = qs.results[module]
            results[module] = res
        else:
            results[module] = pd.DataFrame(columns=["symbol"])

    results["symbol"] = list(
        set(
            sum(
                (
                    results[module]["symbol"].drop_duplicates().to_list()
                    for module in results
                ),
                [],
            )
        )
    )

    return results


def quote_summary(
    symbols: str | list[str], modules: str | list[str] | None = None, *args, **kwargs
):
    return asyncio.run(
        quote_summary_async(symbols=symbols, modules=modules, *args, **kwargs)
    )
