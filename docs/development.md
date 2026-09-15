# Development

Use `uv sync` with Python 3.12+. The development environment pins PPy 0.3.1
within the package's 0.3.x requirement. The LLVM extra is needed by the current
PPy CLI import path, even for an external backend.

```sh
bash scripts/check.sh
```

This gate runs Ruff lint/format verification, Pyright, Pylint, strict pytest,
the installed emission example, strict MkDocs, package builds, and Twine
metadata validation. Temporary pytest and distribution paths are unique and
inside `build/`. The example checks emitted Rust against the committed golden
and cleans its temporary directory.

## Furiosa compiler

Use the baseline-supported Linux environment with Rust
`nightly-2026-05-01` and `cargo-furiosa-opt` 0.6.0 available in PATH:

```sh
bash scripts/furiosa_check.sh
```

The gate generates canonical physical IR and invokes the installed
`ppy build ... --backend furiosa` CLI. It requires a manifest, a nonempty device
binary, and a schedule. An unavailable compiler fails this gate. Windows can
run emission and software tests; real SDK builds require the supported Linux
environment, such as a configured WSL environment with its own Python venv.

## Hardware correctness

```sh
bash scripts/rngd_check.sh
```

This currently exits nonzero with an explicit unsupported-suite message.
Compilation of the broadcast has been demonstrated; its outputs have not been
validated on RNGD. Official MOA tests apply to generated competition kernels,
which this package does not yet provide.
