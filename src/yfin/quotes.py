import pandas as pd
from parallel_requests import parallel_requests

from .constants import URLS


class Quotes:
    _URL = URLS["quotes"]
    _drop_columns = [
        "language",
        "region",
        "typeDisp",
        "customPriceAlertConfidence",
        "triggerable",
        "longName",
        "quoteSourceName",
        "messageBoardId",
        "esgPopulated",
        "exchangeTimezoneShortName",
        "gmtOffSetMilliseconds",
        "sourceInterval",
        "tradeable",
        "priceHint",
        "fiftyTwoWeekRange",
        "underlyingSymbol",
        "openInterest",
    ]
    _date_columns = [
        "firstTradeDateMilliseconds",
        "regularMarketTime",
        "earningsTimestamp",
        "earningsTimestampStart",
        "earningsTimestampEnd",
        "newListingDate",
        "exchangeTransferDate",
        "dividendDate",
        "ipoExpectedDate",
        "postMarketTime",
        "preMarketTime",
    ]

    def __init__(self, symbols: str | list | tuple):
        if isinstance(symbols, str):
            symbols = [symbols]
        self._symbols = symbols

    def fetch(
        self,
        symbols: str | list | tuple = None,
        chunk_size: int = 1500,
        *args,
        **kwargs
    ):
        if symbols is not None:
            if isinstance(symbols, str):
                symbols = [symbols]
            self._symbols = symbols

        self._symbol_chunks = self._chunk_symbols(
            symbols=self._symbols, chunk_size=chunk_size
        )
        params = [dict(symbols=_symbols) for _symbols in self._symbol_chunks]

        results = parallel_requests(
            urls=self._URL, params=params, parse_func=self._parse_raw, *args, **kwargs
        )
        if isinstance(results, list):
            results = pd.concat(
                results,
                ignore_index=True,
            )

        self.results = results

    @staticmethod
    def _chunk_symbols(symbols: list, chunk_size: int = 1500) -> list:
        chunked_symbols = [
            ",".join(symbols[i * chunk_size : (i + 1) * chunk_size])
            for i in range(len(symbols) // chunk_size + 1)
        ]
        chunked_symbols = [cs for cs in chunked_symbols if len(cs) > 0]
        return chunked_symbols

    def _parse_raw(self, response: object) -> pd.DataFrame:

        df = pd.DataFrame(response["quoteResponse"]["result"])

        dates = dict.fromkeys(
            [date for date in self._date_columns if date in df.columns], "datetime64[s]"
        )
        if "firstTradeDateMilliseconds" in dates:
            dates.update({"firstTradeDateMilliseconds": "datetime64[ms]"})
        df = df.drop([col for col in self._drop_columns if col in df.columns], axis=1)
        return df.astype(dates)

    def __call__(self, *args, **kwargs):
        self.fetch(*args, **kwargs)
        return self.results
