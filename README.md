# ppy-furiosa

A PPy extension for generating Furiosa RNGD kernels. The first goal is to
reproduce the **official MOA 2026 Gemma 4 baseline** through PPy, preserving its
ABI, layouts, numerical operations, and rounding order before tuning performance.

The reference is
[`micro2026-moa/furiosa-opt-gemma4-12B`](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/tree/850428729c1b9af0c0b86a9cf694e3b4b4486b29),
pinned at `850428729c1b9af0c0b86a9cf694e3b4b4486b29`.

## Status

The current development version is **0.1.0a1**. A physical BF16 broadcast can
be encoded as canonical `.ppyir`, emitted as Rust, and compiled through the
Furiosa SDK into a device binary and schedule. This is the first helper toward
baseline reproduction; the three complete competition kernels and hardware
correctness checks are still pending.

PPy **0.3.1** is currently locked. Its frontend does not carry generic plugin
types and calls into canonical IR, so `.ppy` source compilation is blocked.
Integration will resume after the upstream **0.3.2** fix is available and verified.
The package registers `ppy.plugins`, `ppy.backends`, and the `furiosa-rust`
emit format. Commands use PPy's CLI.

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
| Compiler | `ppy-lang[llvm]==0.3.1` in `uv.lock` and uv constraints |
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

## Emit example

Generate the current physical broadcast example:

```sh
uv run python examples/broadcast/build_ir.py
```

This writes `build/broadcast.ppyir` and `build/broadcast.rs` using the physical
IR builder and emitter API. Its input is a BF16 vector with `H = 3840`. It uses
the baseline's full-width fetch, 256-slice broadcast, and 16-element collection
from
[`broadcast_hidden`](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/850428729c1b9af0c0b86a9cf694e3b4b4486b29/src/device/layout.rs).
The example adds an HBM output containing all 256 copies so the helper can be
compiled independently. The generated Rust is:

```rust
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

axes![Copies = 256, H = 3840];

#[device(chip = 1)]
pub fn broadcast(
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

Compile the saved IR in a configured Linux SDK environment:

```sh
uv run ppy build build/broadcast.ppyir --backend furiosa -o build/compiled
```

The output contains a Rust crate, source map, device binary, schedule JSON, and
`manifest.json`. A successful compilation does not establish hardware correctness.

The source-level interface being targeted is
`uv run ppy emit furiosa-rust kernel.ppy -o kernel.rs`. It requires the upstream
frontend fix and semantic lowering; it is not a working example in this revision.

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

The README's Rust block is checked against freshly emitted output and the golden
fixture. CI runs software checks on Python 3.12, 3.13, and 3.14 and smoke-tests
the built wheel. Hardware validation is pending; `scripts/rngd_check.sh` currently
fails explicitly until its correctness suite is implemented.

Source lives in `src/ppy_furiosa`, tests in `tests`, and generated artifacts in
ignored `build/`. Mapping/tensor metadata and physical IR are separate from Rust
emission and toolchain execution. PPy 0.3.1 registry compatibility is isolated in
`compatibility.py` and must be reviewed when upgrading PPy.

`main` is the default branch and release line; `dev` is the development line.
Both start from the initial repository setup. Create work branches from `dev`,
merge changes by PR into `dev`, and promote releases to `main`. Use concise,
lower-case commit subjects and delete merged work branches. No stable release
has been published.
