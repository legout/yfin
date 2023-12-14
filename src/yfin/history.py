import asyncio

from requests import session
from .utils.datetime import to_timestamp,
import datetime as dt
import pendulum as pdl
import pandas as pd
from yfin.base import Session

from .constants import URLS


class History:
    """Yahoo Finance Hisorical OHCL Data."""

    _BASE_URL = URLS["chart"]

    def __init__(self, symbols: str | list, session: Session | None = None, *args, **kwargs):
        """
        Initializes a new instance of the class.
        
        Args:
            symbols (str | list): The symbols to be initialized.
            session (Session | None): The session to be used for initialization. Defaults to None.

        """
        if isinstance(symbols, str):
            symbols = [symbols]

        self._symbols = symbols
        
        if session is None:
            session = Session(*args, **kwargs)
        self._session = session

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

    ) -> pd.DataFrame | None:
        """
        Fetches data from an API based on the provided parameters.

        Args:
            start (str | dt.datetime | dt.date | pd.Timestamp | pdl.Date | pdl.DateTime | int | float | None, optional): The start date or timestamp for the data. Defaults to None.
            end (str | dt.datetime | dt.date | pd.Timestamp | pdl.Date | pdl.DateTime | int | float | None, optional): The end date or timestamp for the data. Defaults to None.
            period (str | None, optional): The time period for the data. Defaults to None.
            freq (str, optional): The frequency of the data. Defaults to "1d".
            splits (bool, optional): Whether to include data on stock splits. Defaults to True.
            dividends (bool, optional): Whether to include data on dividends. Defaults to True.
            pre_post (bool, optional): Whether to include pre and post market data. Defaults to False.
            adjust (bool, optional): Whether to adjust the data for stock splits. Defaults to False.
            timezone (str, optional): The timezone for the data. Defaults to "UTC".

        Returns:
            pd.DataFrame | None: The fetched data as a DataFrame, or None if no data is found.
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
        results = await self._session.request_async(
            urls=self._url,
            params=self._params,
            parse_func=_parse,
            keys=self._symbols,
            return_type="json",

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
    session: Session | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetches the historical data for the given symbols.

    Args:
        symbols (str | list): The symbols for which to fetch historical data.
        start (str | dt.datetime | None, optional): The start date or datetime for the historical data. Defaults to None.
        end (str | dt.datetime | None, optional): The end date or datetime for the historical data. Defaults to None.
        period (str | None, optional): The period of the historical data. Defaults to None.
        freq (str, optional): The frequency of the historical data. Defaults to "1d".
        splits (bool, optional): Whether to include splits in the historical data. Defaults to True.
        dividends (bool, optional): Whether to include dividends in the historical data. Defaults to True.
        pre_post (bool, optional): Whether to include pre and post market data in the historical data. Defaults to True.
        adjust (bool, optional): Whether to adjust the historical data for dividends and splits. Defaults to False.
        timezone (str, optional): The timezone for the historical data. Defaults to "UTC".
        session (Session | None, optional): The session to use for the historical data. Defaults to None.

    Returns:
        pd.DataFrame: The historical data for the given symbols.
    """

    h = History(symbols=symbols, session=session, *args, **kwargs)
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
    session: Session | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Get historical data for the specified symbols.

    Args:
        symbols (str | list): The symbols for which to retrieve historical data.
        start (str | dt.datetime | None, optional): The start date of the historical data. Defaults to None.
        end (str | dt.datetime | None, optional): The end date of the historical data. Defaults to None.
        period (str | None, optional): The period of the historical data. Defaults to None.
        freq (str, optional): The frequency of the historical data. Defaults to "1d".
        splits (bool, optional): Whether to include splits data. Defaults to True.
        dividends (bool, optional): Whether to include dividends data. Defaults to True.
        pre_post (bool, optional): Whether to include pre and post market data. Defaults to True.
        adjust (bool, optional): Whether to adjust the data for dividends and splits. Defaults to False.
        timezone (str, optional): The timezone to use for the timestamps. Defaults to "UTC".
        session (Session | None, optional): The session to use for the request. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: The historical data for the specified symbols.
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
