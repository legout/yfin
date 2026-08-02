# Exceptions

Every yfin exception inherits from `yfin.YahooError`, so a single `except
YahooError` catches the entire family.

```text
YahooError
├── YahooAuthError
│   ├── YahooCrumbError
│   └── YahooConsentError
├── YahooRateLimitError
├── YahooApiError
├── YahooSymbolError
└── YahooValidationError
```

## `YahooError`

Base class for every yfin exception. Catch this when you want to handle any
yfin failure mode uniformly.

## `YahooAuthError`

Cookie/crumb authentication failed and no strategy succeeded. Subclasses
are usually raised instead.

## `YahooCrumbError`

Yahoo returned a blank, HTML, or JSON-shaped "crumb". Often a sign that a
consent wall was triggered. yfin automatically tries the CSRF fallback
strategy exactly once before letting this escape.

## `YahooConsentError`

The CSRF consent flow couldn't be parsed — for example, the consent form is
missing the `sessionId` or `csrfToken` hidden fields.

## `YahooRateLimitError`

Yahoo responded with HTTP 429 / Too Many Requests. Carries an optional
`retry_after` attribute parsed from the `Retry-After` header:

```python
import time
import yfin

try:
    yfin.quotes(["AAPL"])
except yfin.YahooRateLimitError as exc:
    time.sleep(exc.retry_after or 1.0)
```

The crumb is automatically cleared on a 429 so the next retry starts clean.

## `YahooApiError`

Yahoo returned a structured error payload (`{"finance": {"error": ...}}`).
Carries an optional `code` attribute:

```python
try:
    yfin.quotes(["BAD_SYMBOL"])
except yfin.YahooApiError as exc:
    print(exc.code, exc)
```

Per-symbol failures inside a multi-symbol call are logged at WARNING and
skipped; you only get `YahooApiError` when **every** symbol failed.

## `YahooSymbolError`

A symbol failed validation or normalisation — empty list, invalid character,
or longer than 12 characters. Raised **before** any network call.

## `YahooValidationError`

Caller-supplied parameters are invalid (both `period` and explicit dates,
invalid `interval`, etc.). Raised before any network call.
