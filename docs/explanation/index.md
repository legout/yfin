# Explanation

Explanation is **understanding-oriented**. It discusses the design choices,
the trade-offs, and the rationale behind yfin's behaviour.

- [Architecture](architecture.md) — module map, dependency direction, async
  boundaries
- [Authentication flow](authentication-flow.md) — warmup, crumb, CSRF
  fallback, per-route state
- [Arrow output design](arrow-output-design.md) — why `pyarrow.Table`, how
  types are inferred, the null-row convention
- [Unofficial endpoints and pacing](unofficial-endpoints-and-pacing.md) —
  what "unofficial" means for stability, the conservative pacing decisions
