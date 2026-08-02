# Authentication flow

Yahoo's public web endpoints usually don't require any authentication, but
the JSON API endpoints used by yfin do: a `Cookie` header for the session
and a `crumb` query parameter for CSRF protection.

The auth flow lives in [`yfin.auth`](../reference/api.md). It
adopts the proven two-strategy flow from `yahooquery`, with one important
twist: **cookie/crumb state is scoped per network route** so a crumb
obtained through one proxy is never sent through another.

## The two strategies

| Strategy | When it's used                                     | What it does                                                       |
|----------|----------------------------------------------------|--------------------------------------------------------------------|
| `BASIC`  | First attempt; default.                            | Warmup `finance.yahoo.com` → extract cookies → GET `getcrumb`       |
| `CSRF`   | Once, after `BASIC` failed.                        | Consent form parse → POST `collectConsent` → GET `copyConsent` → GET `getcrumb` |

`BASIC` is enough in most regions. `CSRF` is the fallback for regions
where Yahoo now serves a consent wall before the warmup redirect chain.

## Per-route state

Every call to `YahooClient.get_route()` returns a fresh `YahooRoute`
representing one explicit route (direct or a specific proxy URL). Cookie
and crumb state is keyed by route:

```
YahooAuth._states: dict[YahooRoute, YahooSessionState]
```

A crumb obtained through `proxy1.example` is **never** sent through
`proxy2.example` or through the direct route. This is what lets yfin use
round-robin proxy rotation safely — Yahoo would reject a request that
mixed IP-based session state.

## What "switched" means

`YahooSessionState.switched` is a one-shot flag. After `BASIC` fails,
`YahooAuth` flips it and retries once with `CSRF`. If `CSRF` also fails,
yfin returns the state with `crumb=None` and lets the request proceed —
the v8 chart endpoint works without a crumb, and `getcrumb` is frequently
rate-limited (429) anyway.

A second failure (e.g. on the next call) does **not** retry CSRF a second
time. The state has to be cleared and the auth flow restarted from
scratch.

## The `client.get_json` flow

```
YahooClient.get_json(url, params, route)
│
├─ state = YahooAuth.ensure_auth(route)        # acquire cookie + crumb
├─ if state.crumb: params["crumb"] = state.crumb
├─ resp = await fastreq.request(url, ..., proxy=route.proxy)
├─ if status == 429: raise YahooRateLimitError, clear crumb
├─ json_data = parse JSON from resp
├─ if detect_yahoo_error(json_data):            # crumb error?
│   ├─ if state.can_switch_strategy(route):    # first time only
│   │   state.switch_strategy()
│   │   return self.get_json(url, params, route)   # retry once
│   └─ else: raise YahooApiError
└─ return json_data
```

This loop is what makes yfin self-healing on crumb rot: a stale crumb
causes a 400 / "Crumb validator failed" payload, yfin switches to CSRF,
acquires a fresh cookie+crumb, and retries — all without raising.

## Crumb validation

The crumb response is validated defensively:

- Blank → `YahooCrumbError`
- Starts with `<html` or `<!doctype` → `YahooCrumbError` (consent wall)
- Starts with `{` or `[` → `YahooCrumbError` (Yahoo returned a JSON error
  payload — accepting it would poison every subsequent request)
- Shorter than 2 chars → `YahooCrumbError`

These checks happen **before** the crumb is cached. The goal is to never
let a bad crumb survive a single request.

## What's next

- [Architecture](architecture.md) — how the auth module fits with the
  providers.
- [Handle errors and rate limits](../how-to/handle-errors-and-rate-limits.md)
  — the caller-visible side of the same flow.
