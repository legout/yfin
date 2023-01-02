import numpy as np
import pandas as pd
from parallel_requests import parallel_requests

from ..constants import URLS


class Search:
    _URL1 = URLS["search"]
    _URL2 = URLS["searchAssist"]

    def __init__(
        self,
        query: str | list,
    ):
        """Symbol search endpoint for Yahoo Finance.

        Args:
            query (str): search query.

        """
        self._query = [query] if isinstance(query, str) else query

    def fetch1(
        self, quotes_count: int = 10, news_count: int = -1, *args, **kwargs
    ) -> pd.DataFrame:
        """Fetch data.
        Args:
            quotes_count (int, optional): max number of quote results. Defaults to 10.
            news_count (int, optional): max number of news results. Defaults to -1.
        Returns:
            pd.DataFrame: results.
        """

        self._quotes_count = quotes_count
        self._news_count = news_count

        params = [
            dict(q=query, quotesCount=self._quotes_count, newsCount=self._news_count)
            for query in self._query
        ]

        results = parallel_requests(
            url=self._URL1,
            params=params,
            key=self._query,
            headers=None,
            parse_func=self._parse_func1,
            method="GET",
            *args,
            **kwargs
        )

        # results = pd.concat(results)

        return results

    def fetch2(self, *args, **kwargs):
        params = {
            "device": "console",
            "returnMeta": "true",
        }
        url = [self._URL2 + query for query in self._query]
        results = parallel_requests(
            url=url,
            params=params,
            key=self._query,
            headers=None,
            parse_func=self._parse_func2,
            method="GET",
            *args,
            **kwargs
        )

        # results = pd.concat(results)

        return results

    def _parse_func1(self, key, result):
        """Parse results from search request."""
        columns = [
            "symbol",
            "longname",
            "exchange",
            "exchDisp",
            "typeDisp",
            "prevName",
            "nameChangeDate",
            "prevTicker",
            "tickerChangeDate",
        ]
        results = list()
        if "quotes" in result:
            quotes = result["quotes"]
            if len(quotes) > 1:
                for quote in quotes:
                    # quote["query"] = key
                    results.append(quote)
                results = pd.DataFrame(results)
            elif len(quotes) == 1:
                results = quotes[0]
                # results["query"] = key
                results = pd.Series(results).to_frame().T
            else:
                results = pd.DataFrame()

            avaiable_columns = [
                col for col in columns if col in results.columns.tolist()
            ]
            missing_columns = [
                col for col in columns if col not in results.columns.tolist()
            ]
            results = results[avaiable_columns].rename(
                {"exchDisp": "exchange_name", "typeDisp": "type", "longname": "name"},
                axis=1,
            )

            if len(missing_columns) > 0:
                results[missing_columns] = np.NaN
                results[missing_columns] = results[missing_columns].astype(str)

        else:
            results = pd.DataFrame(columns=columns)

        return results

    def _parse_func2(self, key, result):
        "Parse results from search request"
        if "data" in result:
            columns = result["data"]["suggestionMeta"]
            data = result["data"]["items"]
            if len(data) == 0:
                results = pd.DataFrame(
                    columns=columns
                )  # .rename({"typeDisp":"type", "exchange"})
            else:
                results = pd.DataFrame(data)

            results = results.drop("type", axis=1).rename(
                {"exchDisp": "exchange_name", "typeDisp": "type", "exch": "exchange"},
                axis=1,
            )
        else:
            results = pd.DataFrame(
                columns=["symbol", "name", "exchange", "echange_name", "type"]
            )

        return results

    def __call__(self, search_assist=1, *args, **kwargs):
        """Fetch data"""
        if search_assist == 1:
            return self.fetch1(*args, **kwargs)
        else:
            return self.fetch2(*args, **kwargs)


def search(
    query: str,
    search_assist=1,
    quotes_count: int = 10,
    news_count: int = -1,
    *args,
    **kwargs
) -> pd.DataFrame:
    """Symbol search yahoo finance for assests.

    Args:
        query (str): search query
        quotes_count (int, optional): max number of quote results. Defaults to 5.
        news_count (int, optional): max number of news results. Defaults to -1.

    Returns:
        pd.DataFrame: search results
    """
    s = Search(query=query)
    if search_assist == 1:
        return s(
            search_assist=1,
            quotes_count=quotes_count,
            news_count=news_count,
            *args,
            **kwargs
        )
    else:
        return s(search_assist=2, *args, **kwargs)
