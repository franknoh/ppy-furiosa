# Contributing

Use Python 3.12+ and `uv sync`. Keep source under `src/ppy_furiosa`, tests under
`tests`, and documentation under `docs`. The package version has one source,
`src/ppy_furiosa/version.py`.

Before submitting a change, run:

```sh
bash scripts/check.sh
```

Changes to generated Rust also require `bash scripts/furiosa_check.sh` in a
configured Linux SDK environment. Record the command, compiler version, and
artifact result. Hardware behavior requires a real correctness run; the current
hardware gate deliberately fails until that suite exists.

Preserve the boundary between PPy canonical IR, plugin-owned semantics, physical
lowering, and backend emission. Do not recover discarded source by parsing it
again in the backend. Reject unsupported operations explicitly. Test malformed
IR as well as valid examples, and update golden Rust intentionally.

Behavior changes include tests, documentation, and a changelog entry. Keep pull
requests focused, with concise descriptions of the change, reason, and exact
validation. Never describe static schedule estimates as hardware measurements.
