# ppy-furiosa

A Furiosa RNGD compiler extension for PPy 0.3.x, currently supporting a verified
physical broadcast pipeline and Rust DSL emission. The package registers a PPy
plugin, backend API 1 implementation, and the `furiosa-rust` format.

The working input is canonical `.ppyir`. A broadcast copies one bf16 vector
from HBM to DM, broadcasts across 256 slices, and writes 256 copies to HBM.
Generated broadcast Rust has compiled with `cargo-furiosa-opt` 0.6.0 and produced
a device binary and schedule. Hardware correctness has not been established.

## Development

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required:

```sh
uv sync
uv run ppy doctor
uv run python scripts/doc_examples.py
```

Create a physical IR artifact with the installed Python API:

```python
from pathlib import Path

from ppy_compiler.ir.codec import write
from ppy_furiosa.physical import broadcast_module, make_registry

write(broadcast_module(), Path("broadcast.ppyir"), make_registry())
```

Emit Rust without the SDK, or build with the Linux Furiosa toolchain:

```sh
uv run ppy-furiosa emit-ir broadcast.ppyir -o broadcast.rs
uv run ppy build broadcast.ppyir --backend furiosa -o build/broadcast
```

Enable project integration through `pyproject.toml`:

```toml
[tool.ppy.plugins.furiosa]
enabled = true

[tool.ppy.backends.furiosa]
target = "rngd"
```

## Current boundary

PPy 0.3.1's canonical frontend does not lower generic plugin types or calls to
custom IR. Consequently, source-level tensor semantics, MOA kernel generation,
and tuning are not implemented. The [backend diagnosis](docs/backend.md)
identifies the upstream connection required. The
[MOA baseline audit](docs/moa2026.md) records the official contract.

Run `bash scripts/check.sh` for the software gate and
`bash scripts/furiosa_check.sh` for a real SDK compile. The hardware gate
`bash scripts/rngd_check.sh` currently fails explicitly because its correctness
suite is not implemented. This release does not complete the MOA milestones.
