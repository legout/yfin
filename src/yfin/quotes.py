import asyncio

import pandas as pd
from parallel_requests import parallel_requests_async

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

    async def fetch(
        self,
        symbols: str | list | tuple = None,
        chunk_size: int = 1500,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch quotes for given symbols.

        Args:
            symbols (str | list): Symbols.
            chunk_size (int, optional): Chunk size of symbols for each request. Defaults to 1000.

        Returns:
            pd.DataFrame: Quotes.
        """

        def _parse(response: object) -> pd.DataFrame:
            df = pd.DataFrame(response["quoteResponse"]["result"])

            dates = dict.fromkeys(
                [date for date in self._date_columns if date in df.columns],
                "datetime64[s]",
            )
            if "firstTradeDateMilliseconds" in dates:
                dates.update({"firstTradeDateMilliseconds": "datetime64[ms]"})
            df = df.drop(
                [col for col in self._drop_columns if col in df.columns], axis=1
            )

            return df.astype(dates)

        def _chunk_symbols(symbols: list, chunk_size: int = 1500) -> list:
            chunked_symbols = [
                ",".join(symbols[i * chunk_size : (i + 1) * chunk_size])
                for i in range(len(symbols) // chunk_size + 1)
            ]
            chunked_symbols = [cs for cs in chunked_symbols if len(cs) > 0]

            return chunked_symbols

        if symbols is not None:
            if isinstance(symbols, str):
                symbols = [symbols]
            self._symbols = symbols

        self._symbol_chunks = _chunk_symbols(
            symbols=self._symbols, chunk_size=chunk_size
        )
        params = [dict(symbols=_symbols) for _symbols in self._symbol_chunks]

        results = await parallel_requests_async(
            urls=self._URL, params=params, parse_func=_parse, *args, **kwargs
        )
        if isinstance(results, list):
            results = pd.concat(
                results,
                ignore_index=True,
            )

        self.results = results

    def __call__(self, *args, **kwargs):
        asyncio.run(self.fetch(*args, **kwargs))
        return self.results


async def quotes_async(
    symbols: str | list, chunk_size: int = 1000, *args, **kwargs
) -> pd.DataFrame:
    """Fetch quotes for given symbols.

    Args:
        symbols (str | list): Symbols.
        chunk_size (int, optional): Chunk size of symbols for each request. Defaults to 1000.

    Returns:
        pd.DataFrame: Quotes.
    """
    q = Quotes(symbols=symbols)
    await q.fetch(chunk_size=chunk_size, *args, **kwargs)

    return q.results


def quotes(
    symbols: str | list, chunk_size: int = 1000, *args, **kwargs
) -> pd.DataFrame:
    """Fetch quotes for given symbols.

    Args:
        symbols (str | list): Symbols.
        chunk_size (int, optional): Chunk size of symbols for each request. Defaults to 1000.

    Returns:
        pd.DataFrame: Quotes.
    """
    return asyncio.run(quotes_async(symbols=symbols, chunk_size=chunk_size, **kwargs))
