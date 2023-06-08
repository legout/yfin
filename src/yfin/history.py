import asyncio
import datetime as dt
from zoneinfo import ZoneInfo

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
        start: str | dt.datetime | dt.date | pd.Timestamp | int | float | None = None,
        end: str | dt.datetime | dt.date | pd.Timestamp | int | float | None = None,
        period: str | None = None,
        freq: str = "1d",
        splits: bool = True,
        dividends: bool = True,
        pre_post: bool = False,
        adjust: bool = False,
        timezone: str = "UTC",
        *args,
        **kwargs
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

            res = response["chart"]["result"][0]

            try:
                timestamp = (
                    pd.Series(res["timestamp"]).astype("datetime64[s]").rename("time")
                )
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

                    if "dividends" in res["events"]:
                        dividends = (
                            pd.DataFrame(res["events"]["dividends"].values())
                            .astype({"date": "datetime64[s]"})
                            .rename({"date": "time"}, axis=1)
                        )

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
                history["time"] = (
                    history["time"].dt.tz_localize("UTC").dt.tz_convert(timezone)
                )
                if freq.lower() in {"1d", "5d", "1wk", "1mo", "3mo"}:
                    history["time"] = history["time"].dt.date

            except Exception:
                history = None
            return history

        url = [self._BASE_URL + symbol for symbol in self._symbols]

        params = {}
        # handle period depending on given period, start, end
        if not start and not period:
            period = "ytd"

        if start:
            if isinstance(start, str):
                start = (
                    dt.datetime.fromisoformat(start)
                    .replace(tzinfo=ZoneInfo(timezone))
                    .timestamp()
                )

            elif isinstance(start, pd.Timestamp):
                start = start.timestamp()
            elif isinstance(start, dt.datetime|dt.date):
                start = start.replace(tzinfo=ZoneInfo(timezone)).timestamp()

            if not end:
                end = dt.datetime.utcnow().timestamp()
            else:
                if isinstance(end, str):
                    end = (
                        dt.datetime.fromisoformat(end)
                        .replace(tzinfo=ZoneInfo(timezone))
                        .timestamp()
                    )

                elif isinstance(end, pd.Timestamp):
                    end = end.timestamp()
                elif isinstance(end, dt.datetime):
                    end = end.replace(tzinfo=ZoneInfo(timezone)).timestamp()

            params = dict(period1=int(start), period2=int(end))

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

        # fetch results
        results = await parallel_requests_async(
            urls=url,
            params=params,
            parse_func=_parse,
            keys=self._symbols,
            *args,
            **kwargs
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
    **kwargs
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
        **kwargs
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
    **kwargs
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
            **kwargs
        )
    )
