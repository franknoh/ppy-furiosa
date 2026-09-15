# ppy-furiosa

A PPy extension for generating Furiosa RNGD kernels. The first goal is to
reproduce the **official MOA 2026 Gemma 4 baseline** through PPy, preserving its
ABI, layouts, numerical operations, and rounding order before tuning performance.

The reference is
[`micro2026-moa/furiosa-opt-gemma4-12B`](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/tree/850428729c1b9af0c0b86a9cf694e3b4b4486b29),
pinned at `850428729c1b9af0c0b86a9cf694e3b4b4486b29`.

## Status

The current development version is **0.1.0a1**, using **PPy 0.3.2**. The package
registers `ppy.plugins`, `ppy.backends`, and the `furiosa-rust` emit format.
Commands use PPy's CLI.

PPy 0.3.2 provides the custom-type and operation frontend bridge, project-scoped
dialect registries, and optional SDK checks for text emission. The extension
uses these APIs; the earlier PPy frontend blocker and global registry workaround
no longer apply. The first supported computation is a BF16 broadcast. Complete
MOA kernels, tuning, and hardware correctness validation are still pending.

## Setup

Use [uv](https://docs.astral.sh/uv/) from the repository root:

```sh
uv sync --locked
uv run ppy doctor
```

| Setting | Version / location |
| --- | --- |
| Development Python | `3.12.13` in `.python-version` |
| Supported Python | `>=3.12` in `pyproject.toml` |
| Compiler | `ppy-lang[llvm]==0.3.2` in `uv.lock` and uv constraints |
| Package version | `src/ppy_furiosa/version.py` |
| Furiosa SDK | `cargo-furiosa-opt` and `furiosa-opt-std` `0.6.0` |
| Generated Rust toolchain | `nightly-2026-05-01` |

`uv sync --locked` installs the package, LLVM support, and development tools in
`.venv`. Rust emission works without the SDK. Device compilation requires the
SDK's supported Linux environment; Windows development can use a separate WSL
environment for compilation.

The repository's `pyproject.toml` enables integration:

```toml
[tool.ppy.plugins.furiosa]
enabled = true

[tool.ppy.backends.furiosa]
target = "rngd"
```

## `kernel.ppy` → Rust

Save the source below as `kernel.ppy` in this project, then emit Rust:

```sh
uv run ppy emit furiosa-rust kernel.ppy -o kernel.rs
```

The same source is checked in at `examples/broadcast/kernel.ppy`. Shapes and
element formats use PPy's annotation metadata; writable destinations use
`fx.MutableTensor`:

```python
from typing import Annotated

import ppy
import ppy_furiosa as fx

Input = Annotated[fx.Tensor, ppy.Shape(3840), ppy.DType("bf16")]
Output = Annotated[fx.MutableTensor, ppy.Shape(256, 3840), ppy.DType("bf16")]


def broadcast(x: Input, out: Output) -> None:
    value = fx.broadcast(x, copies=256)
    fx.store(out, value)
```

Emission uses the official baseline's full-width fetch, 256-slice broadcast, and
16-element collection from
[`broadcast_hidden`](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/850428729c1b9af0c0b86a9cf694e3b4b4486b29/src/device/layout.rs).
An HBM output exposes all 256 copies for independent compilation.

```rust
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

axes![Copies = 256, H = 3840];

#[device(chip = 1)]
pub fn kernel_broadcast(
    ctx: &mut Context,
    x: &HbmTensor<bf16, m![1], m![H]>,
    out: &mut HbmTensor<bf16, m![1], m![Copies, H]>,
) {
    let v0: DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![H]> = x.to_dm(&mut ctx.tdma);
    let v1: DmTensor<bf16, m![1], m![1 # 2], m![Copies], m![H]> = ctx.main
        .begin(v0.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Copies], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    v1.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}
```

## MOA 2026 baseline reproduction

The first complete target is `ops::decoder_feedforward`, followed by the two
sliding-attention entry points. Preserve each official function's name,
signature, return type, and module path. Generated integration must stay within
the competition's allowed `src/device/` and `src/ops.rs` changes.

| Target | Baseline computation | Official test tolerance `(atol, rtol)` |
| --- | --- | --- |
| `decoder_feedforward` | Pre-RMSNorm, FP4 up/gate projections, exact-erf GeGLU, down projection, post-RMSNorm, residual, layer scalar | `(0.01, 0.01)` |
| `sliding_attention_output` | FP8 output projection, post-RMSNorm, residual | `(0.05, 0.01)` |
| `sliding_project_qkv` | Input RMSNorm, FP8 Q/K/V projections, head norms, RoPE, KV cache updates | `(0.04, 0.01)` |

Reference dimensions include `H = 3840`, `L = 15360`, 8 sliding KV heads,
2 queries per KV head, head width 256, and cache length 1024. Reproduction must
preserve the supplied scale conventions, BF16 cast boundaries, reduction order,
mapping expressions, and byte-based cache/RoPE offsets.

Completion requires generated kernels to compile with the pinned SDK and pass
the official
[`tests/test_kernels.rs`](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/850428729c1b9af0c0b86a9cf694e3b4b4486b29/tests/test_kernels.rs)
checks on RNGD for every required output. Record hardware correctness and timing
separately from static schedule estimates. Layout tuning follows baseline
reproduction.

## Development

```sh
bash scripts/check.sh          # lint, types, tests, README emission, package checks
bash scripts/furiosa_check.sh  # real SDK compile; requires Linux + Furiosa SDK
```

Run the checked-in source directly, or save its lowered IR. Device compilation
requires a configured Linux SDK environment:

```sh
uv run ppy emit furiosa-rust examples/broadcast/kernel.ppy -o build/kernel.rs
uv run ppy emit ir examples/broadcast/kernel.ppy -o build/kernel.ppyir
uv run ppy build examples/broadcast/kernel.ppy --backend furiosa -o build/compiled
```

The build retains a Rust crate, source map, device binary, schedule JSON, and
`manifest.json`. The README's source and Rust blocks are checked against the
actual `ppy emit` result and golden fixture. CI runs software checks on Python
3.12, 3.13, and 3.14 and smoke-tests
the built wheel. Hardware validation is pending; `scripts/rngd_check.sh` currently
fails explicitly until its correctness suite is implemented.

Source lives in `src/ppy_furiosa`, tests in `tests`, and generated artifacts in
ignored `build/`. Mapping/tensor metadata and physical IR are separate from Rust
emission and toolchain execution. Dialects are registered per project;
`compatibility.py` contains static typing adapters for public IR attribute APIs.

`main` is the default branch and release line; `dev` is the development line.
Both start from the initial repository setup. Create work branches from `dev`,
merge changes by PR into `dev`, and promote releases to `main`. Use concise,
lower-case commit subjects and delete merged work branches. No stable release
has been published.
