import pandas as pd
from parallel_requests import parallel_requests


class History:

    _BASE_URL = "https://query1.finance.yahoo.com/v7/finance/chart/"

    def __init__(self, symbols: str | list):
        if isinstance(symbols, str):
            symbols = [symbols]

        self._symbols = symbols

    def fetch(
        self,
        period="ytd",
        interval="1d",
        splits=True,
        dividends=True,
        pre_post=True,
        adjust=False,
        timezone="UTC",
        *args,
        **kwargs
    ):

        url = [self._BASE_URL + symbol for symbol in self._symbols]

        params = dict(
            range=period,
            interval=interval,
            events=",".join(["div" * dividends, "split" * splits]),
            close="adjusted" if adjust else "unadjusted",
            includePrePost="true" if pre_post else "false",
        )

        results = parallel_requests(
            url=url,
            params=params,
            parse_func=self._parse_func,
            key=self._symbols,
            *args,
            **kwargs
        )

        results = pd.concat(
            {k: results[k] for k in results if results[k] is not None}, names=["symbol"]
        )

        results["time"] = results["time"].dt.tz_localize("UTC").dt.tz_convert(timezone)

        self.results = results

    def _parse_func(self, key, response):
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

    def __call__(self, *args, **kwargs):
        self.fetch(*args, **kwargs)
        return self.results
