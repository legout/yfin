# API reference

Every public symbol yfin exports, generated from the source docstrings. Use
the navigation tree to browse by module.

::: yfin
    options:
      members: true
      show_source: false
      show_root_full_path: false
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      docstring_style: google
      summary: true

## Notes

- The package is imported as `yfin`; the distribution on PyPI is
  `yfin-client`.
- All async functions have a sync wrapper that calls `asyncio.run`. The sync
  wrapper raises `RuntimeError` when called from a running event loop.
