import pandas as pd
from ..constants import COUNTRIES
from .utils import validate
from async_requests import async_requests
import asyncio
from string import ascii_lowercase, digits
from itertools import product


class Lookup:
    _URL = "https://query1.finance.yahoo.com/v1/finance/lookup"

    async def search(
        self,
        query: str | list,
        type_: str | list = "equity",
        country: str = "united states",
        *args,
        **kwargs
    ) -> pd.DataFrame:
        async def parse_func(key, response):
            res = pd.DataFrame(response["finance"]["result"][0]["documents"])
            res["query"] = key
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

        res = await async_requests(
            url=self._URL,
            params=params,
            key=query,
            parse_func=parse_func,
            *args,
            **kwargs
        )
        if isinstance(res, list):
            res = pd.concat(res, ignore_index=True)
        return res

    async def lookup(
        self,
        query_length: int = 2,
        type_: str | list = "equity",
        country: str = "united states",
        *args,
        **kwargs
    ):

        letters = list(ascii_lowercase)
        numbers = list(digits)
        samples = letters + numbers + [".", "-"]

        queries = [
            "".join(q) for q in list(product(*[samples for n in range(query_length)]))
        ]
        res = await self.search(
            query=queries, type_=type_, country=country, *args, **kwargs
        )

        if isinstance(res, list):
            res = pd.concat(res)
        return res


def lookup_search(
    query: str | list,
    type_: str | list = "equity",
    country: str = "united states",
    *args,
    **kwargs
) -> pd.DataFrame:
    """_summary_

    Args:
        query (str | list): _description_
        type_ (str | list): _description_
        country (str, optional): _description_. Defaults to "united states".

    Returns:
        pd.DataFrame: _description_
    """

    lu = Lookup()
    return asyncio.run(
        lu.search(query=query, type_=type_, country=country, *args, **kwargs)
    )


def lookup(
    query_length: int,
    type_: str | list,
    country: str = "united states",
    *args,
    **kwargs
) -> pd.DataFrame:
    """_summary_

    Args:
        query_length int: _description_
        type_ (str | list): _description_
        country (str, optional): _description_. Defaults to "united states".

    Returns:
        pd.DataFrame: _description_
    """

    lu = Lookup()
    return asyncio.run(
        lu.lookup(
            query_length=query_length, type_=type_, country=country, *args, **kwargs
        )
    )


def download(
    max_combination_length: int = 2,
    types: str = "equity",
    limits_per_host: int = 50,
    semaphore: int = 25,
    use_random_proxy: bool = False,
    verbose: bool = True,
    validate_: bool = True,
) -> pd.DataFrame:
    """Generates all possible combinations from ascii letters, numers, "." and "="
    with a length up to `max_query_length` and fetches the results from the
    yahoo finance symbol lookup endpoint.

    Args:
        max_combination_length (int, optional): maximum combination length . Defaults to 2.
        types (str, optional): Can be anyone or a combination of `equity, mutualfund, etf,
            index, future, currency, cryptocurrency`. Defaults to "equity".
        limits_per_host (int, optional):  Is used to limit the number of parallel requests.
            Should be a value between 10 and 100.. Defaults to 50.
        semaphore (int, optional): Is used to limit the number of parallel requests.
            Should be between smaller than `limits-per-host`.. Defaults to 25.
        use_random_proxy (bool, optional):
            Use this flag to use a random proxy for each request. Currently a list of free proxies is used.
            Defaults to False.
        verbose (bool, optional): Wheter to show a progressbar or not. Defaults to True.
        validate_ (bool, optional): Run a finally validation of the downloaded symbols. Defaults to True.

    Returns:
        pd.DataFrame: symbols
    """

    # lu = Lookup()

    query_lengths = range(1, max_combination_length + 1)
    types = types.split(",")

    results = pd.DataFrame()
    for type_ in types:
        for query_length in query_lengths:
            res_ = lookup(
                query_length=query_length,
                type_=type_,
                limits_per_host=limits_per_host,
                semaphore=semaphore,
                use_random_proxy=use_random_proxy,
                verbose=verbose,
            )
            results = pd.concat(
                [results, res_.drop_duplicates(subset=["symbol", "exchange"])],
                ignore_index=True,
            )

    results = results.rename(
        {"shortName": "name", "quoteType": "type", "industryName": "industry"}, axis=1
    ).drop_duplicates(subset=["symbol", "exchange"])[
        ["symbol", "name", "exchange", "type", "industry"]
    ]
    if validate_:
        validation = validate(
            results["symbol"].tolist(),
            max_symbols=750,
            limits_per_host=limits_per_host,
            semaphore=semaphore,
            verbose=verbose,
        ).reset_index()

        results = results.merge(validation, on=["symbol"])

    return results
