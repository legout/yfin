import asyncio
from itertools import product
from string import ascii_lowercase, digits

import pandas as pd

from ..base import Session
from ..constants import COUNTRIES, URLS


class Lookup:
    _URL = URLS["lookup"]

    def __init__(self, session: Session | None = None, *args, **kwargs):
        """
        Initializes a new instance of the class.

        Args:
            session (Session | None): An optional session object. If not provided,
                a new session will be created using the provided arguments and keyword arguments.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        if session is None:
            session = Session(*args, **kwargs)
        self._session = session

    async def search(
        self,
        query: str | list,
        type_: str | list = "equity",
        country: str = "united states",
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Search for data based on a query or a list of queries.

        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            type_ (Union[str, List[str]], optional): The type or list of types of data to search for.
                Defaults to "equity".
            country (str, optional): The country to search in. Defaults to "united states".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the search results.
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

        results = await self._session.request(
            urls=self._URL,
            params=params,
            # keys=query,
            parse_func=_parse,
            return_type="json",
            *args,
            **kwargs,
        )
        if isinstance(results, list):
            results = pd.concat(results)

        if results.shape[0] > 0:
            renames = {
                k: v
                for k, v in {
                    "shortName": "name",
                    "quoteType": "type",
                    "industryName": "industry",
                }.items()
                if k in results.columns
            }

            results = results.rename(renames, axis=1).drop_duplicates(
                subset=["symbol", "name", "exchange"]
            )

            columns = [
                col
                for col in results.columns
                if col in ["symbol", "name", "exchange", "industry", "type"]
            ]
            results = results[columns]

        return results

    async def lookup(
        self,
        query_length: int = 2,
        type_: str | list = "equity",
        country: str = "united states",
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Performs a lookup operation based on the specified query parameters and returns the
        search results as a pandas DataFrame.

        Args:
            query_length (int, optional): The length of each query to generate. Defaults to 2.
            type_ (str | list, optional): The type of search results to retrieve. Defaults to "equity".
            country (str, optional): The country to limit the search results to. Defaults to "united states".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The search results as a pandas DataFrame.
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
    session: Session | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Asynchronously performs a lookup search based on the provided query, type, and country.

    Args:
        query (str | list): The search query or queries to perform. Can be a single string or a list of strings.
        type_ (str | list, optional): The type or types of search to perform. Defaults to "equity".
        country (str, optional): The country to perform the search in. Defaults to "united states".
        session (Session | None, optional): The session to use for the lookup. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        pd.DataFrame: The search results as a pandas DataFrame.
    """

    lu = Lookup(session=session, *args, **kwargs)
    return await lu.search(query=query, type_=type_, country=country)


def lookup_search(
    query: str | list,
    type_: str | list = "equity",
    country: str = "united states",
    session: Session | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Perform a lookup search based on the given query, type, and country.

    Args:
        query (str | list): The query string or list of query strings to search for.
        type_ (str | list, optional): The type or list of types to filter the search by. Defaults to "equity".
        country (str, optional): The country to filter the search by. Defaults to "united states".
        session (Session | None, optional): The session object to use for the lookup. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: The resulting DataFrame from the search.
    """

    lu = Lookup(session=session, *args, **kwargs)
    return asyncio.run(lu.search(query=query, type_=type_, country=country))


async def lookup_async(
    query_length: int,
    type_: str | list,
    country: str = "united states",
    session: Session | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Asynchronously looks up data based on the given query length, type, country.

    Args:
        query_length (int): The length of the query.
        type_ (str | list): The type of the query.
        country (str, optional): The country for the lookup. Defaults to "united states".
        session (Session | None, optional): The session for the lookup. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: The lookup result as a pandas DataFrame.
    """

    lu = Lookup(session=session, *args, **kwargs)
    return await lu.lookup(
        query_length=query_length,
        type_=type_,
        country=country,
    )


def lookup(
    query_length: int,
    type_: str | list,
    country: str = "united states",
    session: Session | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """
    Lookup function to perform a query for a given query length, type, and country.

    Args:
        query_length (int): The length of the query.
        type_ (str | list): The type of query.
        country (str, optional): The country for the query. Defaults to "united states".
        session (Session | None, optional): The session object. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: The result of the query as a DataFrame.
    """

    lu = Lookup(session=session, *args, **kwargs)
    return asyncio.run(
        lu.lookup(
            query_length=query_length, type_=type_, country=country, *args, **kwargs
        )
    )
