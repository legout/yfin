import asyncio

import pandas as pd
from parallel_requests import parallel_requests_async

from ..constants import URLS


async def validate_async(symbol: str | list, max_symbols=1000, *args, **kwargs):
    """Validation of give symbols. True means the given symbol is a valida
    symbol in the yahoo finance database.

    Args:
        symbol (str | list): symbol
        max_symbols (int, optional): number if symbols included into one request. Defaults to 1000.
    """

    def _parse(response):
        return pd.Series(response["symbolsValidation"]["result"][0]).rename("valid")

    if isinstance(symbol, str):
        symbol = [symbol]
    url = URLS["validation"]
    symbol_ = [
        symbol[i * max_symbols : (i + 1) * max_symbols]
        for i in range(len(symbol) // max_symbols + 1)
    ]
    params = [{"symbols": ",".join(s)} for s in symbol_]

    results = await parallel_requests_async(urls=url, params=params, parse_func=_parse, *args, **kwargs)
    if isinstance(results, list):
        results = pd.concat(results).reset_index().rename({"index": "symbol"}, axis=1)
    elif isinstance(results, pd.Series):
        results = results.reset_index().rename({"index": "symbol"}, axis=1)

    return results


def validate(symbol: str | list, max_symbols=1000, *args, **kwargs):
    """Validation of give symbols. True means the given symbol is a valida
    symbol in the yahoo finance database.

    Args:
        symbol (str | list): symbol
        max_symbols (int, optional): number if symbols included into one request. Defaults to 1000.
    """

    return asyncio.run(validate_async(symbol=symbol, max_symbols=max_symbols, *args, **kwargs))
