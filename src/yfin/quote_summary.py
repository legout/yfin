from datetime import datetime
from .constants import URLS, QUOTE_SUMMARY_COLUMNS, ALL_MODULES
from yahoo.utils import camel_to_snake, snake_to_camel
from async_requests import async_requests
import asyncio
import pandas as pd
import numpy as np

# try:
#     from findata import get_symbols

#     init = False
# except Exception as e:
#     print(e, "Initializing")
#     init = True


def _convert_date(date: int | list) -> datetime | list:
    if isinstance(date, list):
        if isinstance(date[0], int):
            date = [pd.to_datetime(_date, unit="s") for _date in date]
        else:
            date = [pd.to_datetime(_date) for _date in date]
    elif isinstance(date, int):
        date = pd.to_datetime(date, unit="s")
    else:
        date = pd.to_datetime(date)
    return date


class QuoteSummary:

    _BASE_URL = URLS["quoteSummary"]

    def __init__(
        self,
        symbol: str | tuple | list,
        modules: str | tuple | list = [],
    ):
        if isinstance(symbol, str):
            symbol = [symbol]
        self.symbol = symbol
        if isinstance(modules, str):
            modules = [modules]

        _modules = [
            module for module in snake_to_camel(modules) if module in ALL_MODULES.keys()
        ]
        self._modules = camel_to_snake(_modules)
        self._symbol = dict.fromkeys(self._modules, [])
        self.results = dict()

    async def fetch(
        self, modules: str | list | None = None, parse: bool = True, *args, **kwargs
    ):
        # self.symbol = symbol

        self._url = [self._BASE_URL + symbol for symbol in self.symbol]

        if modules is not None:
            if isinstance(modules, str):
                modules = [modules]

            _modules = [
                module
                for module in snake_to_camel(modules)
                if module in ALL_MODULES.keys()
            ]
            self._modules.extend(camel_to_snake(_modules))
            self._symbol.update(dict.fromkeys(_modules, []))

        if self._modules is None:
            self._modules = camel_to_snake(list(ALL_MODULES.keys()))

        self._modules = sorted(set(self._modules))

        if hasattr(self, "_results_raw"):
            _modules = [
                module
                for module in self._modules
                if not await self._module_alread_fetchresult(module)
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

            results = await async_requests(
                url=self._url,
                params=params,
                key=self._symbol,
                parse_func=self._parse_raw if parse else None,
                *args,
                **kwargs,
            )
            
            # remove empy results
            self._removed_symbols = [symbol for symbol in results if results[symbol] is None]
            self._symbol = [symbol for symbol in results if results[symbol] is not None]
            results = {symbol:results[symbol] for symbol in self._symbol}

            # if len(results) > 1:
    
            #     results = {
            #         k: v for result in results for k, v in result.items() if len(v) > 0
            #     }
                
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

    async def _parse_raw(self, key: str, response: object) -> dict:
        if response["quoteSummary"]["result"] is None:
            result = {key: None}
        else:
            res = response["quoteSummary"]["result"][0]
            modules = list(res.keys())
            for module in modules:

                if not module == camel_to_snake(module):
                    res[camel_to_snake(module)] = res[module]
                    res.pop(module)
                if module not in res:
                    res[module] = None
                    
            result = {key: res}

        return result

    async def _module_alread_fetchresult(self, module: str) -> bool:
        if not hasattr(self, "_results_raw"):
            return False
        else:
            return module in self._results_raw[self._symbol[0]].keys()

    async def _module_already_formattresult(self, module: str) -> bool:
        if not hasattr(self, "results"):
            return False
        else:
            return module in self.results

    @staticmethod
    async def _extract_raw(results: dict):

        if "maxAge" in results:
            results.pop("maxAge", None)

        results_ = dict()
        for k in results:
            if isinstance(results[k], dict):
                if "raw" in results[k]:
                    results_[k] = results[k]["raw"]
                elif results[k] == {}:
                    results_[k] = np.nan
                else:
                    results_[k] = results[k]
            else:
                results_[k] = results[k]

        return results_

    @staticmethod
    async def _to_dataframe(results: list, convert_dates: list = None) -> pd.DataFrame:
        results = pd.DataFrame(results)
        if convert_dates is not None:
            for col in convert_dates:
                if col in results:
                    # pass
                    results[col] = results[col].apply(_convert_date)
            # results = results.astype(dict.fromkeys(convert_dates, "datetime64[s]"))

        results.columns = camel_to_snake(results.columns.tolist())

        return results

    @staticmethod
    async def _to_series(results: list, convert_dates: list = None) -> pd.DataFrame:
        results = pd.Series(results)
        if convert_dates is not None:
            for idx in convert_dates:
                if idx in results:
                    # pass
                    results[idx] = _convert_date(results[idx])
                    # results[idx] = pd.to_datetime(results[idx], unit="s")

        results.index = camel_to_snake(results.index.tolist())

        return results

    async def _format_earnings_trends(self, result: dict):

        formatted_results = dict()

        formatted_results["period"] = result["period"]
        formatted_results["date"] = result["endDate"]

        earnings_estimate = await self._extract_raw(result["earningsEstimate"])
        earnings_estimate_keys = [
            "earnings_estimate_" + idx for idx in earnings_estimate.keys()
        ]
        earnings_estimate = dict(
            zip(earnings_estimate_keys, earnings_estimate.values())
        )
        formatted_results.update(earnings_estimate)

        revenue_estimate = await self._extract_raw(result["revenueEstimate"])
        revenue_estimate_keys = [
            "revenue_estimate_" + idx for idx in revenue_estimate.keys()
        ]
        revenue_estimate = dict(zip(revenue_estimate_keys, revenue_estimate.values()))
        formatted_results.update(revenue_estimate)

        eps_revisions = await self._extract_raw(result["epsRevisions"])
        eps_revisions_keys = ["eps_revisions_" + idx for idx in eps_revisions.keys()]
        eps_revisions = dict(zip(eps_revisions_keys, eps_revisions.values()))
        formatted_results.update(eps_revisions)

        eps_trend = await self._extract_raw(result["epsTrend"])
        eps_trend_keys = ["eps_trend_" + idx for idx in eps_trend.keys()]
        eps_trend = dict(zip(eps_trend_keys, eps_trend.values()))
        formatted_results.update(eps_trend)

        return formatted_results

    async def format_earnings_trends(self, **kwargs):
        key = ALL_MODULES["earningsTrend"]["key"]
        convert_dates = ALL_MODULES["earningsTrend"]["convert_dates"]

        results = {
            symbol: (
                await self._to_dataframe(
                    [
                        await self._format_earnings_trends(res)
                        for res in self._results_raw[symbol]["earnings_trend"][key]
                    ],
                    convert_dates=convert_dates,
                )
            ).dropna(subset=["date"])
            for symbol in self._results_raw
        }
        self.results["earnings_trend"] = results

    async def format_results(self, module: str):

        module_ = snake_to_camel(module)
        key = ALL_MODULES[module_]["key"]
        convert_dates = ALL_MODULES[module_]["convert_dates"]
        convert = ALL_MODULES[module_]["convert"]
        if module == "earnings_trend":
            await self.format_earnings_trends()
        else:
            if convert == "df":
                results = {
                    symbol: await self._to_dataframe(
                        [
                            await self._extract_raw(rr)
                            for rr in (
                                self._results_raw[symbol][module][key]
                                if key is not None
                                else self._results_raw[symbol][module]
                            )
                        ],
                        convert_dates=convert_dates,
                    )
                    for symbol in self._results_raw
                }

            else:
                results = {
                    symbol: await self._to_series(
                        await (
                            self._extract_raw(self._results_raw[symbol][module][key])
                            if key is not None
                            else self._extract_raw(self._results_raw[symbol][module])
                        ),
                        convert_dates=convert_dates,
                    )
                    for symbol in self._results_raw
                }

            self.results[module] = results

    async def _balance_sheet_history(
        self, quarterly: bool = False, **kwargs
    ) -> pd.DataFrame:
        module = "balance_sheet_history"
        if quarterly:
            module += "_quarterly"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _calendar_events(self, **kwargs) -> pd.DataFrame:
        module = "calendar_events"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _cashflow_statement_history(
        self, quarterly: bool = False, **kwargs
    ) -> pd.DataFrame:
        module = "cashflow_statement_history"
        if quarterly:
            module += "_quarterly"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _default_key_statistics(self, **kwargs) -> pd.DataFrame:
        module = "default_key_statistics"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _earnings(self, **kwargs) -> pd.DataFrame:
        module = "earnings"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _earnings_history(self, **kwargs) -> pd.DataFrame:
        module = "earnings_history"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _earnings_trend(self, **kwargs) -> pd.DataFrame:
        module = "earnings_trend"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _financial_data(self, **kwargs) -> pd.DataFrame:
        module = "financial_data"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _fund_ownership(self, **kwargs) -> pd.DataFrame:
        module = "fund_ownership"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _income_statement_history(
        self, quarterly: bool = False, **kwargs
    ) -> pd.DataFrame:
        module = "income_statement_history"
        if quarterly:
            module += "_quarterly"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _insider_transactions(self, **kwargs) -> pd.DataFrame:
        module = "insider_transactions"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _insider_holders(self, **kwargs) -> pd.DataFrame:
        module = "insider_holders"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _institution_ownership(self, **kwargs) -> pd.DataFrame:
        module = "institution_ownership"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _major_holders_breakdown(self, **kwargs) -> pd.DataFrame:
        module = "major_holders_breakdown"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _net_share_purchase_activity(self, **kwargs) -> pd.DataFrame:
        module = "net_share_purchase_activity"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _price(self, **kwargs) -> pd.DataFrame:
        module = "price"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _quote_type(self, **kwargs) -> pd.DataFrame:
        module = "quote_type"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _recommendation_trend(self, **kwargs) -> pd.DataFrame:
        module = "recommendation_trend"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _summary_detail(self, **kwargs) -> pd.DataFrame:
        module = "summary_detail"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    async def _summary_profile(self, **kwargs) -> pd.DataFrame:
        module = "summary_profile"
        if not await self._module_alread_fetchresult(module):
            await self.fetch(modules=module, **kwargs)

        if not await self._module_already_formattresult(module):
            await self.format_results(module=module)

        return self.results[module]

    @property
    def balance_sheet_history(self, **kwargs):
        return asyncio.run(self._balance_sheet_history(quarterly=False, **kwargs))

    @property
    def balance_sheet_history_quarterly(self, **kwargs):
        return asyncio.run(self._balance_sheet_history(quarterly=True, **kwargs))

    @property
    def calendar_events(self, **kwargs):
        return asyncio.run(self._calendar_events(**kwargs))

    @property
    def calendar_events(self, **kwargs):
        return asyncio.run(self._calendar_events(**kwargs))

    @property
    def calendar_events(self, **kwargs):
        return asyncio.run(self._calendar_events(**kwargs))

    @property
    def cashflow_statement_history(self, **kwargs):
        return asyncio.run(self._cashflow_statement_history(quarterly=False, **kwargs))

    @property
    def cashflow_statement_history_quarterly(self, **kwargs):
        return asyncio.run(self._cashflow_statement_history(quarterly=True, **kwargs))

    @property
    def default_key_statistics(self, **kwargs):
        return asyncio.run(self._default_key_statistics(**kwargs))

    @property
    def earnings_history(self, **kwargs):
        return asyncio.run(self._earnings_history(**kwargs))

    @property
    def earnings_trend(self, **kwargs):
        return asyncio.run(self._earnings_trend(**kwargs))

    @property
    def fund_ownership(self, **kwargs):
        return asyncio.run(self._fund_ownership(**kwargs))

    @property
    def _income_statement_history(self, **kwargs):
        return asyncio.run(self._income_statement_history(quarterly=False, **kwargs))

    @property
    def _income_statement_history_quarterly(self, **kwargs):
        return asyncio.run(self._income_statement_history(quarterly=True, **kwargs))

    @property
    def insider_holders(self, **kwargs):
        return asyncio.run(self._insider_holders(**kwargs))

    @property
    def insider_transactions(self, **kwargs):
        return asyncio.run(self._insider_transactions(**kwargs))

    @property
    def major_holders_breakdown(self, **kwargs):
        return asyncio.run(self._major_holders_breakdown(**kwargs))

    @property
    def net_share_purchase_activity(self, **kwargs):
        return asyncio.run(self._net_share_purchase_activity(**kwargs))

    @property
    def price(self, **kwargs):
        return asyncio.run(self._price(**kwargs))

    @property
    def quote_type(self, **kwargs):
        return asyncio.run(self._quote_type(**kwargs))

    @property
    def recommendation_trend(self, **kwargs):
        return asyncio.run(self._recommendation_trend(**kwargs))

    @property
    def summary_detail(self, **kwargs):
        return asyncio.run(self._summary_detail(**kwargs))

    @property
    def summary_profile(self, **kwargs):
        return asyncio.run(self._summary_profile(**kwargs))

    # async def _parse_result(
    #     self, result_raw: dict, _filter: str, convert: str, convert_dates: list
    # ) -> pd.Series | pd.DataFrame:

    #     if _filter is not None:
    #         result_raw = result_raw[_filter]

    #     if convert == "df":
    #         res = []
    #         for _self in result_raw:
    #             _res = {}
    #             for k in _self:
    #                 if isinstance(_self[k], dict):
    #                     if "raw" in _self[k]:
    #                         _res[k] = _self[k]["raw"]
    #                     elif len(_self[k]) == 0:
    #                         _res[k] = None
    #                     else:
    #                         _res[k] = _self[k]
    #                 else:
    #                     _res[k] = _self[k]
    #             # res.append(_res)
    #             res = pd.concat([res, _res], ignore_index=True)

    #         res = pd.DataFrame(res)
    #         # date_columns = [col for col in res.columns if col in convert_dates]
    #         # res[date_columns] = res[date_columns].astype("datetime64[s]")
    #         res.columns = [
    #             camel_to_snake(col.replace("+", "p").replace("-", "m"))
    #             for col in res.columns
    #         ]
    #         if "max_age" in res.columns:
    #             res = res.drop("max_age", axis=1)
    #         if "price_hint" in res.index:
    #             res = res.drop("price_hint", axis=1)

    #     else:

    #         if _filter is not None:
    #             result_raw = result_raw[_filter]

    #         res = pd.Series(result_raw)
    #         # date_indices = [idx for idx in res.index if idx in convert_dates]
    #         # for date_idx in date_indices:
    #         #    res[date_idx] = pd.to_datetime(res[date_idx], unit="s")

    #         res.index = [
    #             camel_to_snake(idx.replace("+", "p").replace("-", "m"))
    #             for idx in res.index
    #         ]
    #         if "max_age" in res.index:
    #             res = res.drop("max_age")
    #         if "price_hint" in res.index:
    #             res = res.drop("price_hint")

    #         res = res.replace("Infinity", np.nan).apply(pd.to_numeric, errors="ignore")

    #     return res

    # async def _parse_earnings_history(self, earnings_history_raw) -> pd.Series:
    #     earnings_history = await self._parse_result(
    #         earnings_history_raw, _filter="history", convert="df", convert_dates=[]
    #     )
    #     earnings_history = earnings_history.drop("quarter", axis=1)
    #     earnings_history = earnings_history.rename(
    #         {"surprisePercent": "epsSurprise"}, axis=1
    #     )
    #     earnings_history = earnings_history.melt(
    #         id_vars="period"
    #     )  # .set_index('period').stack()
    #     index = (
    #         (earnings_history["variable"] + earnings_history["period"])
    #         .str.replace("-", "_-")
    #         .str.replace("+", "p")
    #         .str.replace("-", "m")
    #     )
    #     earnings_history = earnings_history["value"]
    #     earnings_history.index = index
    #     earnings_history.name = "earnings_history"
    #     return earnings_history

    # async def _parse_earnings_trend(self, earnings_trend_raw) -> pd.Series:
    #     earnings_estimate = pd.Series()
    #     revenue_estimate = pd.Series()
    #     for _earnings_trend_raw in earnings_trend_raw["trend"]:
    #         period = _earnings_trend_raw["period"]
    #         end_date = _earnings_trend_raw["endDate"]
    #         growth = (
    #             _earnings_trend_raw["growth"]["raw"]
    #             if "raw" in _earnings_trend_raw["growth"]
    #             else np.nan
    #         )
    #         _earnings_estimate = pd.Series(
    #             {
    #                 k: _earnings_trend_raw["earningsEstimate"][k]["raw"]
    #                 if _earnings_trend_raw["earningsEstimate"][k] != {}
    #                 else np.nan
    #                 for k in _earnings_trend_raw["earningsEstimate"]
    #             }
    #         )
    #         if _earnings_estimate["growth"] == np.nan:
    #             _earnings_estimate["growth"] = growth
    #         _earnings_estimate.index = _earnings_estimate.index.map(
    #             lambda x: f"earnings_estimate_{period}_" + x
    #         )
    #         # earnings_estimate = earnings_estimate.append(_earnings_estimate)
    #         earnings_estimate = pd.concat(
    #             [earnings_estimate, _earnings_estimate], ignore_index=True
    #         )
    #         earnings_estimate.index = [
    #             camel_to_snake(idx.replace("+", "p").replace("-", "m"))
    #             for idx in earnings_estimate.index
    #         ]

    #         _revenue_estimate = pd.Series(
    #             {
    #                 k: _earnings_trend_raw["revenueEstimate"][k]["raw"]
    #                 if _earnings_trend_raw["revenueEstimate"][k] != {}
    #                 else np.nan
    #                 for k in _earnings_trend_raw["revenueEstimate"]
    #             }
    #         )
    #         _revenue_estimate.index = _revenue_estimate.index.map(
    #             lambda x: f"revenue_estimate_{period}_" + x
    #         )
    #         # revenue_estimate = revenue_estimate.append(_revenue_estimate)
    #         revenue_estimate = pd.concat(
    #             [revenue_estimate, _revenue_estimate], ignore_index=True
    #         )
    #         revenue_estimate.index = [
    #             camel_to_snake(idx.replace("+", "p").replace("-", "m"))
    #             for idx in revenue_estimate.index
    #         ]

    #     return earnings_estimate, revenue_estimate

    # async def _parse_insider_transactions(self, insider_transactions_raw):
    #     # insider_transactions = await self._parse_result(
    #     #    insider_transactions_raw,
    #     #    _filter="transactions",
    #     #    convert="df",
    #     #    convert_dates=[],
    #     # )

    #     # def _sale_or_purchase(x):
    #     #    if "sale" in x.lower():
    #     #        return "Sale"
    #     #    elif "purchase" in x.lower():
    #     #        return "Purchase"
    #     #    else:
    #     #        return None

    #     # insider_transactions["transaction"] = insider_transactions[
    #     #    "transaction_text"
    #     # ].apply(lambda x: _sale_or_purchase(x))

    #     return insider_transactions_raw

    # async def _parse_calendar_events(self, calendar_events_raw):
    #     calendar_events = pd.Series(calendar_events_raw["earnings"])

    #     if "earningsDate" in calendar_events:
    #         if isinstance(calendar_events["earningsDate"], list):
    #             if len(calendar_events["earningsDate"]) > 0:
    #                 calendar_events["earningsDate"] = calendar_events["earningsDate"][0]
    #             else:
    #                 calendar_events["earningsDate"] = 0
    #         elif isinstance(calendar_events["earningsDate"], int):
    #             calendar_events["earningsDate"] = calendar_events["earningsDate"]
    #         else:
    #             calendar_events["earningsDate"] = 0
    #     calendar_events.index = [camel_to_snake(idx) for idx in calendar_events.index]
    #     return calendar_events

    # async def parse_results(self, key, results_raw: dict) -> dict:
    #     symbol = key  # list(results_raw.keys(**kwargs)[0]

    #     results = {symbol: {}}
    #     try:

    #         _results_raw = results_raw["quoteSummary"]["result"][0]  # [symbol]

    #         for module in self._modules:
    #             if module in _results_raw:
    #                 result_raw = _results_raw[module]
    #                 _filter = ALL_MODULES[module]["_filter"]
    #                 convert = ALL_MODULES[module]["convert"]
    #                 convert_dates = ALL_MODULES[module]["convert_dates"]

    #                 if module == "earningsHistory":
    #                     results[symbol][
    #                         camel_to_snake(module)
    #                     ] = await self._parse_earnings_history(result_raw)
    #                 elif module == "earningsTrend":
    #                     (
    #                         earnings_estimates,
    #                         revenue_estimates,
    #                     ) = await self._parse_earnings_trend(result_raw)
    #                     results[symbol]["earnings_estimates"] = earnings_estimates
    #                     results[symbol]["revenue_estimates"] = revenue_estimates
    #                 elif module == "calendarEvents":
    #                     results[symbol][
    #                         camel_to_snake(module)
    #                     ] = await self._parse_calendar_events(result_raw)
    #                 else:
    #                     results[symbol][
    #                         camel_to_snake(module)
    #                     ] = await self._parse_result(
    #                         result_raw,
    #                         _filter=_filter,
    #                         convert=convert,
    #                         convert_dates=convert_dates,
    #                     )

    #     except TypeError:
    #         pass
    #     return results


# async def _fetch_quote_summary(
#     symbols: str | int | list | None= None,
#     exchanges: str | int | list | None= None,
#     countries: str | list | None = None,
#     modules: list = None,
#     limits_per_host: int = 25,
#     sema: int = 30,
#     use_proxy: bool = True,
#     *args,
#     **kwargs,
# ) -> list:

#     symbols = get_symbols(
#         symbol_ids=symbols, exchanges=exchanges, countries=countries, yahoo=True
#     )

#     conn = aiohttp.TCPConnector(limit_per_host=limits_per_host, verify_ssl=False)
#     sema = asyncio.Semaphore(sema)

#     self = QuoteSummary()

#     async with aiohttp.ClientSession(connector=conn) as session:

#         tasks = [
#             asyncio.create_task(
#                 self.fetch(
#                     symbol=symbol,
#                     modules=modules,
#                     session=session,
#                     sema=sema,
#                     use_proxy=use_proxy,
#                     *args,
#                     **kwargs,
#                 )
#             )
#             for symbol in symbols
#         ]

#         results = [
#             await task
#             for task in progressbar.progressbar(
#                 asyncio.as_completresult(tasks), max_value=len(symbols)
#             )
#         ]
#     # results = [res for res in results if res is not None]
#     results = {k: v for result in results for k, v in result.items() if len(v) > 0}

#     return results


def fetch_quote_summary(
    symbols: str | int | list | None = None,
    exchanges: str | int | list | None = None,
    countries: str | list | None = None,
    modules: list = None,
    *args,
    **kwargs,
) -> list:
    if "use_proxy" in kwargs:
        use_proxy = kwargs.pop("use_proxy")
    else:
        use_proxy = True
    if init:
        symbols = symbols
    else:
        symbols = get_symbols(
            symbol_ids=symbols, exchanges=exchanges, countries=countries, yahoo=True
        )
    self = QuoteSummary(symbol=symbols, modules=modules)
    return self(use_proxy=use_proxy, *args, **kwargs)


def fetch_quote_summary_basic(
    symbols: str | int | list | None = None,
    exchanges: str | int | list | None = None,
    countries: str | list | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    modules = [
        "summary_detail",
        "calendar_events",
        "default_key_statistics",
        "earnings_history",
        "earnings_trend",
        "financial_data",
        "major_holders_breakdown",
        "netSharePurchaseActivity",
        "quote_type",
        "summary_profile",
        "price",
    ]

    results = fetch_quote_summary(
        symbols=symbols,
        exchanges=exchanges,
        countries=countries,
        modules=modules,
        *args,
        **kwargs,
    )

    results = pd.concat(
        [pd.concat(results[k]).droplevel(0) for k in results],  # .drop_duplicates()
        keys=results.keys(),
        axis=0,
    )
    results = results[~results.index.duplicatresult()].unstack()

    results = results.reset_index(drop=True)
    results = results[[col for col in QUOTE_SUMMARY_COLUMNS if col in results.columns]]

    # .drop(
    #     ["fax", "phone", "zip", "company_officers", "address1", "address2"], axis=1
    # )
    results = results.apply(pd.to_numeric, errors="ignore")
    percent_columns = [col for col in results.columns if "percent" in col]
    results[percent_columns] = results[percent_columns] * 100
    results["time"] = pd.Timestamp.now().date()

    dtypes = {
        "short_name": str,
        "volume": float,
        "category": str,
        "fund_family": str,
        "legal_type": str,
        "fax": str,
        "phone": str,
        "zip": str,
        "address1": str,
        "adress2": str,
        "address3": str,
        "gmt_off_set_milliseconds": int,
    }
    results = results.astype(
        {col: dtypes[col] for col in dtypes if col in results.columns}
    )
    results = results.rename({"first_trade_date_epoch_utc": "ipo_date"}, axis=1)

    for col in [
        "date_short_interest",
        "ex_dividend_date",
        "ipo_date",
        "last_dividend_date",
        "last_split_date",
        "last_fiscal_year_end",
        "earnings_date",
        "shares_short_previous_month_date",
        "next_fiscal_year_end",
        "most_recent_quarter",
    ]:
        if col in results.columns:
            results[col] = pd.to_datetime(results[col], unit="s")

    # results["time"] = results["time"].astype("datetime64[ns]")
    results.drop_duplicates(subset=["symbol", "short_name", "industry", "sector"])
    # results = results.rename({"index":"symbol"}, axis=1)
    return results
