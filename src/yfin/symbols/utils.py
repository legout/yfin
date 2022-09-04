import pandas as pd
from async_requests import requests
from ..constants import URLS


def validate(symbol: str | list, max_symbols=1000, **kwargs):
    """Validation of give symbols. True means the given symbol is a valida
    symbol in the yahoo finance database.

    Args:
        symbol (str | list): symbol
        max_symbols (int, optional): number if symbols included into one request. Defaults to 1000.
    """

    async def parse_json(json):
        return pd.Series(json["symbolsValidation"]["result"][0])

    if isinstance(symbol, str):
        symbol = [symbol]
    url = URLS["validation"]
    symbol_ = [
        symbol[i * max_symbols : (i + 1) * max_symbols]
        for i in range(len(symbol) // max_symbols + 1)
    ]
    params = [{"symbols": ",".join(s)} for s in symbol_]

    res = pd.concat(
        requests(url=url, params=params, parse_func=parse_json, **kwargs)
    ).rename("valid")
    res.index.names = ["symbol"]

    return res


# def fetch_symbol_info(
#     isins: pd.DataFrame = None,
#     symbols: str | int | list | pd.DataFrame | None = None,
#     exchanges: str | int | list | None = None,
#     countries: str | list | None = None,
#     strip_symbol: bool = False,
#     *args,
#     **kwargs
# ) -> pd.DataFrame:
#     """Function uses the `Search` class to requests infos of symbols.
#     symbols can be given as icker, exchange or countries.

#     Args:
#         isins (str | list | None, optional): ISINs. Defaults to None.
#         symbols (str | int | list | None, optional): symbols. Defaults to None.
#         exchanges (str | int | list | None, optional): uses all symbols of the given exchanges . Defaults to None.
#         countries (str | list | None, optional): uses all symbols of the given countries. Defaults to None.

#     Returns:
#         pd.DataFrame: symbol_info
#     """
#     if "use_proxy" in kwargs:
#         use_proxy = kwargs.pop("use_proxy")
#     else:
#         use_proxy = True

#     if isins is None:
#         isins = get_isin(
#             symbols=symbols,
#             exchanges=exchanges,
#             countries=countries,
#             with_symbol_ids=True,
#         )

#     if symbols is None:
#         symbols = get_symbols(
#             symbol_ids=symbols,
#             exchanges=exchanges,
#             coutries=countries,
#             with_symbol_ids=True,
#         )
#     else:
#         if not isinstance(symbols, pd.DataFrame):
#             symbols = get_symbols(
#                 symbol_ids=symbols,
#                 exchanges=exchanges,
#                 coutries=countries,
#                 with_symbol_ids=True,
#             )

#     # if use_isin:
#     isins_ = isins.dropna()

#     s = Search(
#         query=isins_["isin"],
#         quotes_count=1,
#         news_count=0,
#     )
#     symbol_info = s(use_proxy=use_proxy, *args, **kwargs)
#     symbol_info["symbol_id"] = symbol_info["query"].replace(
#         isins.set_index("isin").to_dict()["symbol_id"]
#     )
#     symbol_info = symbol_info.rename({"query": "isin"}, axis=1)
#     if strip_symbol:
#         symbol_info["symbol"] = symbol_info["symbol"].apply(lambda x: x.split(".")[0])
#     # else:
#     #     s = Search(
#     #         query=symbols["symbol"],
#     #         quotes_count=1,
#     #         news_count=0,
#     #     )
#     #     symbol_info = s(use_proxy=use_proxy, *args, **kwargs)
#     #     symbol_info["symbol_id"] = symbol_info["query"].replace(
#     #         symbols.set_index("symbol").to_dict()["symbol_id"]
#     #     )
#     #     symbol_info["isin"] = symbol_info["symbol_id"].replace(
#     #         isins.set_index("symbol_id").to_dict()["isin"]
#     #     )

#     symbol_info["time"] = pd.Timestamp.now().date()

#     return symbol_info


# def lookup(search_term: str, type_: str | None = "equity", **kwargs) -> pd.DataFrame:
#     """Scrapes symbol data from yahoo finance lookup page.

#     Args:
#         search_term (str): Can be any string.
#         type_ (str | None): symbol type. Can be equity, etfs, mutualfund,index,currency

#     Returns:
#         pd.DataFrame: results.
#     """

#     async def _parse_html(html):
#         try:
#             return pd.read_html(html)[0]
#         except:
#             return pd.DataFrame()

#     url = URLS["lookup"]

#     params0 = {"s": search_term, "b": 0, "c": 100}
#     r0 = requests(url=url, params=params0, return_type="text", **kwargs)

#     res0 = asyncio.run(_parse_html(r0))

#     n_stocks_str = re.findall("Stocks \([0-9]*\)", r0)  # ["Stocks (197)"]
#     if len(n_stocks_str) > 0:
#         n_stocks = int(n_stocks_str[0].split("Stocks (")[-1][:-1])

#     else:
#         n_stocks = 0

#     pages = min(100, (n_stocks - 1) // 100 + 1)
#     params = [{"s": search_term, "b": page * 100, "c": 100} for page in range(1, pages)]

#     if len(params) > 0:
#         res = requests(
#             url=url, params=params, return_type="text", parse_func=_parse_html, **kwargs
#         )
#         if isinstance(res, list):
#             res = pd.concat(res)

#         res = pd.concat([res0, res])
#     else:
#         res = res0

#     symbols = (
#         res.drop_duplicates()
#         .rename(
#             {
#                 "Symbol": "symbol",
#                 "Name": "name",
#                 "Last Price": "last_price",
#                 "Industry / Category": "industry",
#                 "Type": "type",
#                 "Exchange": "exchange",
#             },
#             axis=1,
#         )
#         .dropna(subset=["symbol"])
#     )
#     symbols = symbols.drop(0, axis=1)
