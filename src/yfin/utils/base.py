import re
    

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


# def get_yahoo_cookie():
#     #cookie = None

#     user_agent_key = "User-Agent"
#     user_agent_value = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"

#     headers = {user_agent_key: user_agent_value}
#     response = requests.get(
#         "https://finance.yahoo.com", headers=headers, allow_redirects=True
#     )

#     if not response.cookies:
#         raise Exception("Failed to obtain Yahoo auth cookie.")

#     return list(response.cookies)[0]


# def get_yahoo_crumb(cookie):
#     crumb = None

#     user_agent_key = "User-Agent"
#     user_agent_value = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"

#     headers = {user_agent_key: user_agent_value}

#     crumb_response = requests.get(
#         "https://query1.finance.yahoo.com/v1/test/getcrumb",
#         headers=headers,
#         cookies={cookie.name: cookie.value},
#         allow_redirects=True,
#     )
#     crumb = crumb_response.text

#     if crumb is None:
#         raise Exception("Failed to retrieve Yahoo crumb.")

#     return crumb