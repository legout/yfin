import asyncio
from itertools import product
from string import ascii_lowercase, digits

import pandas as pd
from parallel_requests import parallel_requests_async

from ..constants import COUNTRIES, URLS


class Lookup:
    _URL = URLS["lookup"]

    async def search(
        self,
        query: str | list,
        type_: str | list = "equity",
        country: str = "united states",
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """Run query search on "https://query1.finance.yahoo.com/v1/finance/lookup"

        Args:
            query (str | list): Search query.
            type_ (str | list, optional): Asset type. Defaults to "equity".
            country (str, optional): Country. Defaults to "united states".

        Returns:
            pd.DataFrame: Search results.
        """

        def _parse(response):
            res = pd.DataFrame(response["finance"]["result"][0]["documents"])

            return res

        if isinstance(query, str):
            query = [query]
        if isinstance(type_, list):
            type_ = ",".join(type_)

        params = [
            dict(
                formatted="false",
                query=query_,
                type=type_,
                count=10000,
                start=0,
                lang=COUNTRIES[country]["lang"],
                region=COUNTRIES[country]["region"],
                corsDomain=COUNTRIES[country]["corsDomain"],
            )
            for query_ in query
        ]

        results = await parallel_requests_async(
            urls=self._URL,
            params=params,
            # keys=query,
            parse_func=_parse,
            *args,
            **kwargs
        )
        if isinstance(results, list):
            results = pd.concat(results)
        return results

    async def lookup(
        self,
        query_length: int = 2,
        type_: str | list = "equity",
        country: str = "united states",
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """Run query search for `all` queries with length `query_length`.

        Args:
            query_length (int, optional): Query length. Defaults to 2.
            type_ (str | list, optional): Asset type. Defaults to "equity".
            country (str, optional): Country. Defaults to "united states".

        Returns:
            pd.DataFrame: Search results.
        """

        letters = list(ascii_lowercase)
        numbers = list(digits)
        samples = letters + numbers + [".", "-"]

        queries = [
            "".join(q) for q in list(product(*[samples for n in range(query_length)]))
        ]
        results = await self.search(
            query=queries, type_=type_, country=country, *args, **kwargs
        )

        # if isinstance(res, list):
        #    res = pa.concat_tables(res)
        return results


async def lookup_search_async(
    query: str | list,
    type_: str | list = "equity",
    country: str = "united states",
    *args,
    **kwargs
) -> pd.DataFrame:
    """Run query search for `all` queries with length `query_length`.

    Args:
        query_length (int, optional): Query length. Defaults to 2.
        type_ (str | list, optional): Asset type. Defaults to "equity".
        country (str, optional): Country. Defaults to "united states".

    Returns:
        pd.DataFrame: Search results.
    """

    lu = Lookup()
    return await lu.search(query=query, type_=type_, country=country, *args, **kwargs)

def lookup_search(
    query: str | list,
    type_: str | list = "equity",
    country: str = "united states",
    *args,
    **kwargs
) -> pd.DataFrame:
    """Run query search for `all` queries with length `query_length`.

    Args:
        query_length (int, optional): Query length. Defaults to 2.
        type_ (str | list, optional): Asset type. Defaults to "equity".
        country (str, optional): Country. Defaults to "united states".

    Returns:
        pd.DataFrame: Search results.
    """

    lu = Lookup()
    return asyncio.run(lu.search(query=query, type_=type_, country=country, *args, **kwargs))


async def lookup_async(
    query_length: int,
    type_: str | list,
    country: str = "united states",
    *args,
    **kwargs
) -> pd.DataFrame:
    """Run query search for `all` queries with length `query_length`.

    Args:
        query_length (int, optional): Query length. Defaults to 2.
        type_ (str | list, optional): Asset type. Defaults to "equity".
        country (str, optional): Country. Defaults to "united states".

    Returns:
        pd.DataFrame: Search results.
    """

    lu = Lookup()
    return await lu.lookup(
        query_length=query_length, type_=type_, country=country, *args, **kwargs
    )
    
def lookup(
    query_length: int,
    type_: str | list,
    country: str = "united states",
    *args,
    **kwargs
) -> pd.DataFrame:
    """Run query search for `all` queries with length `query_length`.

    Args:
        query_length (int, optional): Query length. Defaults to 2.
        type_ (str | list, optional): Asset type. Defaults to "equity".
        country (str, optional): Country. Defaults to "united states".

    Returns:
        pd.DataFrame: Search results.
    """

    lu = Lookup()
    return asyncio.run(lu.lookup(
        query_length=query_length, type_=type_, country=country, *args, **kwargs
    ))


# def download(
#     max_combination_length: int = 2,
#     types: str = "equity",
#     limits_per_host: int = 50,
#     semaphore: int = 25,
#     use_random_proxy: bool = False,
#     verbose: bool = True,
#     validate_: bool = True,
# ) -> pd.DataFrame:
#     """Generates all possible combinations from ascii letters, numers, "." and "="
#     with a length up to `max_query_length` and fetches the results from the
#     yahoo finance symbol lookup endpoint.

#     Args:
#         max_combination_length (int, optional): maximum combination length . Defaults to 2.
#         types (str, optional): Can be anyone or a combination of `equity, mutualfund, etf,
#             index, future, currency, cryptocurrency`. Defaults to "equity".
#         limits_per_host (int, optional):  Is used to limit the number of parallel requests.
#             Should be a value between 10 and 100.. Defaults to 50.
#         semaphore (int, optional): Is used to limit the number of parallel requests.
#             Should be between smaller than `limits-per-host`.. Defaults to 25.
#         use_random_proxy (bool, optional):
#             Use this flag to use a random proxy for each request. Currently a list of free proxies is used.
#             Defaults to False.
#         verbose (bool, optional): Wheter to show a progressbar or not. Defaults to True.
#         validate_ (bool, optional): Run a finally validation of the downloaded symbols. Defaults to True.

#     Returns:
#         pd.DataFrame: symbols
#     """

#     # lu = Lookup()

#     query_lengths = range(1, max_combination_length + 1)
#     types = types.split(",")

#     results = pd.DataFrame()
#     for type_ in types:
#         for query_length in query_lengths:
#             res_ = lookup(
#                 query_length=query_length,
#                 type_=type_,
#                 limits_per_host=limits_per_host,
#                 semaphore=semaphore,
#                 use_random_proxy=use_random_proxy,
#                 verbose=verbose,
#             )
#             results = pd.concat(
#                 [results, res_.drop_duplicates(subset=["symbol", "exchange"])],
#                 ignore_index=True,
#             )

#     results = results.rename(
#         {"shortName": "name", "quoteType": "type", "industryName": "industry"}, axis=1
#     ).drop_duplicates(subset=["symbol", "exchange"])[
#         ["symbol", "name", "exchange", "type", "industry"]
#     ]
#     if validate_:
#         validation = validate(
#             results["symbol"].tolist(),
#             max_symbols=750,
#             limits_per_host=limits_per_host,
#             semaphore=semaphore,
#             verbose=verbose,
#         ).reset_index()

#         results = results.merge(validation, on=["symbol"])

#     return results
