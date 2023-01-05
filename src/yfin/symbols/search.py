import urllib.parse

import numpy as np
import pandas as pd
from parallel_requests import parallel_requests
from tqdm import tqdm
from ..constants import URLS


class Search:
    _URL1 = URLS["search"]
    _URL2 = URLS["searchAssist"]

    def __init__(self, query: str | list, max_queries: int = 2500):
        """Symbol search endpoint for Yahoo Finance.

        Args:
            query (str): search query.
            max_queries (int): maximum queries in on parallel request.

        """
        if isinstance(query, str):
            query = [query]

        self._queries = [
            query[i * max_queries : (i + 1) * max_queries]
            for i in range(len(query) // max_queries + 1)
            if len(query[i * max_queries : (i + 1) * max_queries]) > 0
        ]

    def _fetch1(
        self, quotes_count: int = 10, news_count: int = -1, *args, **kwargs
    ) -> pd.DataFrame:
        """Fetch data.
        Args:
            quotes_count (int, optional): max number of quote results. Defaults to 10.
            news_count (int, optional): max number of news results. Defaults to -1.
        Returns:
            pd.DataFrame: results.
        """

        def _parse(result):
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
                    {
                        "exchDisp": "exchange_name",
                        "typeDisp": "type",
                        "longname": "name",
                    },
                    axis=1,
                )
                results["type"] = results["type"].str.lower()

                if len(missing_columns) > 0:
                    results[missing_columns] = np.NaN
                    results[missing_columns] = results[missing_columns].astype(str)

            else:
                results = pd.DataFrame(columns=columns)

            return results

        self._quotes_count = quotes_count
        self._news_count = news_count

        params = [
            [
                dict(
                    q=query, quotesCount=self._quotes_count, newsCount=self._news_count
                )
                for query in queries
            ]
            for queries in self._queries
        ]

        results = []
        for i, params_ in enumerate(params):
            results_ = parallel_requests(
                urls=self._URL1,
                params=params_,
                keys=self._queries[i],
                headers=None,
                parse_func=_parse,
                method="GET",
                *args,
                **kwargs
            )

            if isinstance(results_, dict):
                results_ = (
                    pd.concat(results_, names=["query"])
                    .reset_index()
                    .drop("level_1", axis=1)
                )

            results.append(results_)

        return pd.concat(results)

    def _fetch2(self, *args, **kwargs):
        """Fetch 2"""

        def _parse(result):
            "Parse results from search request"
            if "data" in result:
                columns = result["data"]["suggestionMeta"]
                data = result["data"]["items"]
                if len(data) == 0:
                    results = pd.DataFrame(columns=columns)
                else:
                    results = pd.DataFrame(data)

                results = results.drop("type", axis=1).rename(
                    {
                        "exchDisp": "exchange_name",
                        "typeDisp": "type",
                        "exch": "exchange",
                    },
                    axis=1,
                )
                results["type"] = results["type"].str.lower()
            else:
                results = pd.DataFrame(columns=columns)

            return results

        params = {
            "device": "console",
            "returnMeta": "true",
        }
        urls = [
            [self._URL2 + urllib.parse.quote(query) for query in queries]
            for queries in self._queries
        ]

        results = []
        for i, queries in enumerate(self._queries):
            results_ = parallel_requests(
                urls=urls[i],
                params=params,
                keys=queries,
                headers=None,
                parse_func=_parse,
                method="GET",
                *args,
                **kwargs
            )

            if isinstance(results_, dict):
                results_ = (
                    pd.concat(results_, names=["query"])
                    .reset_index()
                    .drop("level_1", axis=1)
                )

            results.append(results_)

        return pd.concat(results)

    def fetch(self, search_assist=1, *args, **kwargs):
        """Fetch data"""
        if search_assist == 1:
            return self._fetch1(*args, **kwargs)
        else:
            return self._fetch2(*args, **kwargs)

    def __call__(self, search_assist=1, *args, **kwargs):
        return self.fetch(search_assist=search_assist, *args, **kwargs)


def search(
    query: str,
    search_assist=1,
    max_queries: int = 2500,
    quotes_count: int = 10,
    news_count: int = -1,
    *args,
    **kwargs
) -> pd.DataFrame:
    """Symbol search yahoo finance for assests.

    Args:
        query (str): Search query
        quotes_count (int, optional): max number of quote results. Defaults to 5.
        news_count (int, optional): max number of news results. Defaults to -1.

    Returns:
        pd.DataFrame: search results
    """
    s = Search(query=query, max_queries=max_queries)
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
