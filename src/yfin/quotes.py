import asyncio

import pandas as pd
import requests
from parallel_requests import parallel_requests_async

from .constants import URLS
from .utils.base import camel_to_snake


class Quotes:
    _URL = URLS["quotes"]
    _drop_columns = [
        "language",
        "region",
        # "typeDisp",
        "customPriceAlertConfidence",
        "triggerable",
        # "longName",
        "quoteSourceName",
        "messageBoardId",
        "esgPopulated",
        # "exchangeTimezoneShortName",
        "gmtOffSetMilliseconds",
        "sourceInterval",
        "tradeable",
        "priceHint",
        "fiftyTwoWeekRange",
        # "underlyingSymbol",
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
    all_fields = [
        "ask",
        "askSize",
        "averageAnalystRating",
        "averageDailyVolume10Day",
        "averageDailyVolume3Month",
        "bid",
        "bidSize",
        "bookValue",
        "currency",
        "corporateActions",
        "displayName",
        "dividendDate",
        "dividendRate",
        "dividendYield",
        "earningsTimestamp",
        "earningsTimestampEnd",
        "earningsTimestampStart",
        "epsCurrentYear",
        "epsForward",
        "epsTrailingTwelveMonths",
        "fiftyDayAverage",
        "fiftyDayAverageChange",
        "fiftyDayAverageChangePercent",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekHighChange",
        "fiftyTwoWeekHighChangePercent",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekLowChange",
        "fiftyTwoWeekLowChangePercent",
        "fiftyTwoWeekRange",
        "financialCurrency",
        "forwardPE",
        "longName",
        "marketCap",
        "messageBoardId",
        "preMarketChange",
        "preMarketChangePercent",
        "preMarketPrice",
        "preMarketTime",
        "postMarketChange",
        "postMarketChangePercent",
        "postMarketPrice",
        "postMarketTime",
        "priceEpsCurrentYear",
        "priceToBook",
        "quantity",
        "regularMarketChange",
        "regularMarketChangePercent",
        "regularMarketDayHigh",
        "regularMarketDayLow",
        "regularMarketDayRange",
        "regularMarketOpen",
        "regularMarketPreviousClose",
        "regularMarketVolume",
        "sharesOutstanding",
        "shortName",
        "trailingAnnualDividendRate",
        "trailingAnnualDividendYield",
        "trailingPE",
        "twoHundredDayAverage",
        "twoHundredDayAverageChange",
        "twoHundredDayAverageChangePercent",
    ]

    def __init__(self, symbols: str | list | tuple):
        if isinstance(symbols, str):
            symbols = [symbols]
        self._symbols = symbols

        self._cookies = self._get_yahoo_cookie()
        self._crumb = self._get_yahoo_crumb(self._cookies)

    @staticmethod
    def _get_yahoo_cookie():
        cookie = None

        user_agent_key = "User-Agent"
        user_agent_value = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"

        headers = {user_agent_key: user_agent_value}
        response = requests.get(
            "https://fc.yahoo.com", headers=headers, allow_redirects=True
        )

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        return list(response.cookies)[0]

    @staticmethod
    def _get_yahoo_crumb(cookie):
        crumb = None

        user_agent_key = "User-Agent"
        user_agent_value = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"

        headers = {user_agent_key: user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    async def fetch(
        self,
        symbols: str | list | tuple = None,
        chunk_size: int = 1500,
        fields: list | None = None,
        *args,
        **kwargs,
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
        fields = self.all_fields if fields is None else fields

        params = [
            dict(symbols=_symbols, crumb=self._crumb, fields=",".join(fields))
            for _symbols in self._symbol_chunks
        ]
        results = await parallel_requests_async(
            urls=self._URL,
            params=params,
            parse_func=_parse,
            cookies={self._cookies.name: self._cookies.value},
            return_type="json",
            *args,
            **kwargs,
        )

        if isinstance(results, list):
            results = pd.concat(
                results,
                ignore_index=True,
            )
        if results is not None:
            results.columns = [camel_to_snake(col) for col in results.columns]

        self.results = results

    def __call__(self, *args, **kwargs):
        asyncio.run(self.fetch(*args, **kwargs))
        return self.results


async def quotes_async(
    symbols: str | list,
    chunk_size: int = 1000,
    fields: list | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Fetch quotes for given symbols.

    Args:
        symbols (str | list): Symbols.
        chunk_size (int, optional): Chunk size of symbols for each request. Defaults to 1000.

    Returns:
        pd.DataFrame: Quotes.
    """
    q = Quotes(symbols=symbols)
    await q.fetch(chunk_size=chunk_size, fields=fields, *args, **kwargs)

    return q.results


def quotes(
    symbols: str | list,
    chunk_size: int = 1000,
    fields: list | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Fetch quotes for given symbols.

    Args:
        symbols (str | list): Symbols.
        chunk_size (int, optional): Chunk size of symbols for each request. Defaults to 1000.

    Returns:
        pd.DataFrame: Quotes.
    """
    return asyncio.run(
        quotes_async(symbols=symbols, chunk_size=chunk_size, fields=fields, **kwargs)
    )
