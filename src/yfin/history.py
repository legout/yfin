import asyncio
from time import time
from .utils.datetime import to_timestamp, datetime_from_string
import datetime as dt
import pendulum as pdl
import pandas as pd
from parallel_requests import parallel_requests_async

from .constants import URLS


class History:
    """Yahoo Finance Hisorical OHCL Data."""

    _BASE_URL = URLS["chart"]

    def __init__(self, symbols: str | list):
        if isinstance(symbols, str):
            symbols = [symbols]

        self._symbols = symbols

    async def fetch(
        self,
        start: str
        | dt.datetime
        | dt.date
        | pd.Timestamp
        | pdl.Date
        | pdl.DateTime
        | int
        | float
        | None = None,
        end: str
        | dt.datetime
        | dt.date
        | pd.Timestamp
        | pdl.Date
        | pdl.DateTime
        | int
        | float
        | None = None,
        period: str | None = None,
        freq: str = "1d",
        splits: bool = True,
        dividends: bool = True,
        pre_post: bool = False,
        adjust: bool = False,
            timezone: str = "UTC",
        *args,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Fetch historical ohcl data from yahoo finance.

        Args:
            start (str | dt.datetime | None, optional): Download start time. Defaults to None.
            end (str | dt.datetime | None, optional): Download end time. Defaults to None.
            period (str | None, optional): Download period. Defaults to None.
                Either use period or start and end to define download period.
                Valid options: 11d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            freq (str, optional): Download frequence. Defaults to "1d".
                Valid options: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            splits (bool, optional): Include splilts into downloaded dataframe. Defaults to True.
            dividends (bool, optional): Include dividends into downloaded dataframe. Defaults to True.
            pre_post (bool, optional): Include data from pre and/or post market. Defaults to True.
            adjust (bool, optional): Auto adjust ohcl data. Defaults to False.
            timezone (str, optional): Timezone used for timestamp. Defaults to "UTC".

        Returns:
            pd.DataFrame: ohcl history.
        """

        def _parse(response):
            splits = pd.DataFrame(columns=["time", "splitRatio"])
            dividends = pd.DataFrame(columns=["time", "amount"])
            adjclose = pd.DataFrame(columns=["adjclose"])
            if "chart" in response:
                res = response["chart"]["result"][0]

                timestamp =pd.Series(res["timestamp"]).apply(pd.to_datetime, unit="s", utc=True).rename("time")

                ohlcv = pd.DataFrame(res["indicators"]["quote"][0])

                if "adjclose" in res["indicators"]:
                    adjclose = pd.DataFrame(res["indicators"]["adjclose"][0])

                if "events" in res:
                    if "splits" in res["events"]:

                        def splitratio_to_float(s):
                            if isinstance(s, str):
                                a, b = s.split(":")
                                return int(a) / int(b)
                            return s

                        splits = (
                            pd.DataFrame(res["events"]["splits"].values())
                            .astype({"date": "datetime64[s]"})
                            .rename({"date": "time"}, axis=1)[["time", "splitRatio"]]
                        )
                        splits["splitRatio"] = splits["splitRatio"].apply(
                            lambda s: splitratio_to_float(s)
                        )
                        splits["time"] =splits["time"].dt.tz_localize("UTC")

                    if "dividends" in res["events"]:
                        dividends = (
                            pd.DataFrame(res["events"]["dividends"].values())
                            .astype({"date": "datetime64[s]"})
                            .rename({"date": "time"}, axis=1)
                        )
                        dividends["time"] =dividends["time"].dt.tz_localize("UTC")

                history = (
                    pd.concat([timestamp, ohlcv, adjclose], axis=1)
                    .merge(splits, on=["time"], how="left")
                    .merge(dividends, on=["time"], how="left")
                    .fillna(0)
                )

                if adjust:
                    history[["open", "high", "low", "close"]] = (
                        history[["open", "high", "low", "close"]]
                        * (history["adjclose"] / history["close"]).values[:, None]
                    )
                if timezone != "UTC":
                    history["time"] = (
                        history["time"].dt.tz_convert(timezone)
                    )
                if freq.lower() in {"1d", "5d", "1wk", "1mo", "3mo"}:
                    history["time"] = history["time"].dt.date

                history = history.replace({"Infinity": "inf", "-Infinity": "-inf"})
                dtypes = {
                    k: v
                    for k, v in {
                        "symbol": str,
                        #"time": "datetime64[s]",
                        "low": float,
                        "high": float,
                        "volume": int,
                        "open": float,
                        "close": float,
                        "adjclose": float,
                        "splitRatio": float,
                        "amount": float,
                    }.items()
                    if k in history.columns
                }
                history = history.astype(dtypes)

            else:
                history = None
            return history

        self._url = [self._BASE_URL + symbol for symbol in self._symbols]

        params = {}
        # handle period depending on given period, start, end
        if not start and not period:
            period = "ytd"

        if start:
            start = to_timestamp(start, timezone=timezone)

            if not end:
                end = int(pdl.now().timestamp())
            else:
                end=to_timestamp(end, timezone=timezone)

            params = dict(period1=start, period2=end)

        if period:
            params = dict(range=period)

        # set params
        params.update(
            dict(
                interval=freq,
                events=",".join(["div" * dividends, "split" * splits]),
                close="adjusted" if adjust else "unadjusted",
                includePrePost="true" if pre_post else "false",
            )
        )
        self._params = params
        # fetch results
        results = await parallel_requests_async(
            urls=self._url,
            params=self._params,
            parse_func=_parse,
            keys=self._symbols,
            return_type="json",
            *args,
            **kwargs,
        )

        # combine results
        if isinstance(results, dict):
            not_none_results = {
                k: results[k] for k in results if results[k] is not None
            }
            if not_none_results:
                results = (
                    pd.concat(
                        {k: results[k] for k in results if results[k] is not None},
                        names=["symbol"],
                    )
                    .reset_index()
                    .drop("level_1", axis=1)
                )
                # replace

                # dtypes
                results = results[
                    [
                        "symbol",
                        "time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjclose",
                        "volume",
                        "amount",
                        "splitRatio",
                    ]
                ]
            else:
                results = None

        self.results = results

    def __call__(self, *args, **kwargs):
        asyncio.run(self.fetch(*args, **kwargs))
        return self.results


async def history_async(
    symbols: str | list,
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    period: str | None = None,
    freq: str = "1d",
    splits: bool = True,
    dividends: bool = True,
    pre_post: bool = True,
    adjust: bool = False,
    timezone: str = "UTC",
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Fetch historical ohcl data from yahoo finance.

    Args:
        start (str | dt.datetime | None, optional): Download start time. Defaults to None.
        end (str | dt.datetime | None, optional): Download end time. Defaults to None.
        period (str | None, optional): Download period. Defaults to None.
            Either use period or start and end to define download period.
            Valid options: 11d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        freq (str, optional): Download frequence. Defaults to "1d".
            Valid options: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        splits (bool, optional): Include splilts into downloaded dataframe. Defaults to True.
        dividends (bool, optional): Include dividends into downloaded dataframe. Defaults to True.
        pre_post (bool, optional): Include data from pre and/or post market. Defaults to True.
        adjust (bool, optional): Auto adjust ohcl data. Defaults to False.
        timezone (str, optional): Timezone used for timestamp. Defaults to "UTC".

    Returns:
        pd.DataFrame: ohcl history.
    """
    h = History(symbols=symbols)
    await h.fetch(
        start=start,
        end=end,
        period=period,
        freq=freq,
        splits=splits,
        dividends=dividends,
        pre_post=pre_post,
        adjust=adjust,
        timezone=timezone,
        *args,
        **kwargs,
    )
    return h.results


def history(
    symbols: str | list,
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    period: str | None = None,
    freq: str = "1d",
    splits: bool = True,
    dividends: bool = True,
    pre_post: bool = True,
    adjust: bool = False,
    timezone: str = "UTC",
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Fetch historical ohcl data from yahoo finance.

    Args:
        start (str | dt.datetime | None, optional): Download start time. Defaults to None.
        end (str | dt.datetime | None, optional): Download end time. Defaults to None.
        period (str | None, optional): Download period. Defaults to None.
            Either use period or start and end to define download period.
            Valid options: 11d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        freq (str, optional): Download frequence. Defaults to "1d".
            Valid options: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        splits (bool, optional): Include splilts into downloaded dataframe. Defaults to True.
        dividends (bool, optional): Include dividends into downloaded dataframe. Defaults to True.
        pre_post (bool, optional): Include data from pre and/or post market. Defaults to True.
        adjust (bool, optional): Auto adjust ohcl data. Defaults to False.
        timezone (str, optional): Timezone used for timestamp. Defaults to "UTC".

    Returns:
        pd.DataFrame: ohcl history.
    """

    return asyncio.run(
        history_async(
            symbols=symbols,
            start=start,
            end=end,
            period=period,
            freq=freq,
            splits=splits,
            dividends=dividends,
            pre_post=pre_post,
            adjust=adjust,
            timezone=timezone,
            *args,
            **kwargs,
        )
    )
