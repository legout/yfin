---
hide:
  - navigation
  - toc
---

# yfin documentation

Compact Python 3.14 Yahoo Finance client that returns `pyarrow.Table` — built
on [fastreq](https://github.com/legout/fastreq)'s `curl_cffi` backend with
browser TLS impersonation.

<div class="grid cards" markdown>

- **Tutorials**

    ---

    Hands-on lessons for new users. Build competence step by step.

    [Start learning →](tutorials/index.md)

- **How-to guides**

    ---

    Goal-oriented recipes for specific tasks. Already productive — just need
    the steps.

    [Browse recipes →](how-to/index.md)

- **Reference**

    ---

    Concise, accurate descriptions of every public symbol, schema, constant,
    and exception.

    [API reference →](reference/index.md)

- **Explanation**

    ---

    Conceptual background: architecture, design choices, the trade-offs behind
    the implementation.

    [Read the why →](explanation/index.md)

</div>

## Quick example

```python
import yfin
from datetime import date

# Batch quotes
quotes = yfin.quotes(
    ["AAPL", "MSFT"],
    fields=["regularMarketPrice", "regularMarketVolume", "currency"],
)

# Historical OHLCV
history = yfin.history(["AAPL", "MSFT"], start=date(2024, 1, 1), end=date(2024, 6, 1))

# Fundamentals timeseries (valuation + market cap, last 4 years)
fundamentals = yfin.fundamentals(["AAPL"], types=yfin.VALUATION_TYPES)

# Quote summary modules
profile = yfin.asset_profile(["AAPL"])
```

All four call sites return `pyarrow.Table` with deterministic schemas.

## Where to go next

| If you want to…                              | Read…                                       |
|----------------------------------------------|---------------------------------------------|
| Fetch your first table end-to-end            | [Tutorials / Getting started](tutorials/getting-started.md) |
| Quote / history / fundamentals in your code  | [How-to guides](how-to/index.md)            |
| Look up a specific function or schema        | [Reference](reference/index.md)             |
| Understand the auth flow or output shape     | [Explanation](explanation/index.md)         |
