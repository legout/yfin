import pandas as pd
from parallel_requests import parallel_requests
import datetime as dt
from .constants import URLS


class History:

    _BASE_URL = URLS["chart"]

    def __init__(self, symbols: str | list):
        if isinstance(symbols, str):
            symbols = [symbols]

        self._symbols = symbols

    def fetch(
        self,
        start: str | dt.datetime | None = None,
        end: str | dt.datetime | None = None,
        period: str | None = None,  
        interval: str = "1d",
        splits: bool = True,
        dividends: bool = True,
        pre_post: bool = True,
        adjust: bool = False,
        timezone: str = "UTC",
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """_summary_

        Args:
            start (str | dt.datetime | None, optional): _description_. Defaults to None.
            end (str | dt.datetime | None, optional): _description_. Defaults to None.
            period (str | None, optional): _description_. Defaults to None.
            interval (str, optional): _description_. Defaults to "1d".
            splits (bool, optional): _description_. Defaults to True.
            dividends (bool, optional): _description_. Defaults to True.
            pre_post (bool, optional): _description_. Defaults to True.
            adjust (bool, optional): _description_. Defaults to False.
            timezone (str, optional): _description_. Defaults to "UTC".

        Returns:
            pd.DataFrame: _description_
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
                        splits = (
                            pd.DataFrame(res["events"]["splits"].values())
                            .astype({"date": "datetime64[s]"})
                            .rename({"date": "time"}, axis=1)[["time", "splitRatio"]]
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

            except:
                history = None
            return history

        url = [self._BASE_URL + symbol for symbol in self._symbols]

        if not start and not period:
            period = "ytd"

        if start:
            start = dt.datetime.fromisoformat(str).timestamp()
            if not end:
                end = dt.datetime.utcnow().timestamp()
            params = dict(period1=start, period2=end)

        if period:
            params = dict(range=period)

        params.update(
            dict(
                interval=interval,
                events=",".join(["div" * dividends, "split" * splits]),
                close="adjusted" if adjust else "unadjusted",
                includePrePost="true" if pre_post else "false",
            )
        )

        results = parallel_requests(
            urls=url,
            params=params,
            parse_func=_parse,
            keys=self._symbols,
            *args,
            **kwargs
        )

        if isinstance(results, list):
            results = pd.concat(
                {k: results[k] for k in results if results[k] is not None},
                names=["symbol"],
            )

            results["time"] = (
                results["time"].dt.tz_localize("UTC").dt.tz_convert(timezone)
            )

        self.results = results

    def __call__(self, *args, **kwargs):
        self.fetch(*args, **kwargs)
        return self.results


def history(
    symbols: str | list,
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    period: str|None = None,
    interval: str = "1d",
    splits: bool = True,
    dividends: bool = True,
    pre_post: bool = True,
    adjust: bool = False,
    timezone: str = "UTC",
    *args,
    **kwargs
) -> pd.DataFrame:
    """Fetch historical ohcl data.

    Args:
        start (str | dt.datetime | None, optional): _description_. Defaults to None.
        end (str | dt.datetime | None, optional): _description_. Defaults to None.
        symbols (str | list): Symbols.
        period (str, optional): History period. 
            Valid options: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. Defaults to "ytd".
        interval (str, optional): Interval between timestamps.
            Valid options: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,maxDefaults to "1d".
        splits (bool, optional): _description_. Defaults to True.
        dividends (bool, optional): _description_. Defaults to True.
        pre_post (bool, optional): _description_. Defaults to True.
        adjust (bool, optional): Adjust . Defaults to False.
        timezone (str, optional): Convert timestamps to given timezone. Defaults to "UTC".

    Returns:
        pd.DataFrame: History

    """
    h = History(symbols=symbols)
    h.fetch(
        period=period,
        interval=interval,
        splits=splits,
        dividends=dividends,
        pre_post=pre_post,
        adjust=adjust,
        timezone=timezone,
        *args,
        **kwargs
    )
    return h.results
