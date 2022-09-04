import requests

import pandas as pd
import re
import random


def _camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def camel_to_snake(name: str | list) -> str | list:
    if isinstance(name, (list, tuple)):
        name = [_camel_to_snake(n) for n in name]
    else:
        name = _camel_to_snake(name)
    return name


def _snake_to_camel(name):
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), name)


def snake_to_camel(name: str | list) -> str | list:
    if isinstance(name, (list, tuple)):
        name = [_snake_to_camel(n) for n in name]
    else:
        name = _snake_to_camel(name)
    return name


def convert_numbers(x):
    abb = ["", "K", "M", "B", "T"]
    div = [1e0, 1e3, 1e6, 1e9, 1e12]
    try:
        x = int(x)
        idx = len(str(x)) // 3
        _abb = abb[idx]
        _div = div[idx]
        return f"{x / _div:.2f} {_abb}"
    except:
        return x
