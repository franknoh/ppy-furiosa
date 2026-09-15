# Backend

The installed distribution registers:

| Entry-point group | Name | Factory/value |
| --- | --- | --- |
| ppy.plugins | furiosa | ppy_furiosa.plugin:create_plugin |
| ppy.backends | furiosa | ppy_furiosa.backend:create_backend |
| ppy.backend-formats | furiosa-rust | furiosa |

The backend implements API 1. Its `furiosa-rust` format is module-scoped and uses
`.rs`. Current emission is available through
`ppy-furiosa emit-ir input.ppyir -o output.rs`; `.ppy` semantic source emission
is unavailable for the reason below.

## Configuration and artifacts

```toml
[tool.ppy.plugins.furiosa]
enabled = true

[tool.ppy.backends.furiosa]
target = "rngd"
cargo = "cargo"
rustc = "rustc"
```

Only `rngd` is accepted. Unknown backend options and empty command names fail.
MOA paths are not backend options. `ppy doctor` reports discovered backend
status; `ppy-furiosa doctor` directly checks cargo, rustc, and the Furiosa command.
Import does not require those executables.

`ppy build input.ppyir --backend furiosa -o build/output` creates one crate per
module, with `src/lib.rs`, `Cargo.toml`, `rust-toolchain.toml`, `source-map.json`,
and per-kernel `device/` output. The root `manifest.json` records package/PPy
versions, backend identity, and generated files. Modules are not concatenated.
Each rebuild invalidates its previous manifest before changing source; only a
successful build publishes a new manifest atomically. Consumers must require
that manifest rather than use leftover files from a failed build.

The crate pins `furiosa-opt-std = "=0.6.0"` and `nightly-2026-05-01`.
Compilation stages fresh artifacts and requires a device binary and schedule.
The compiler's substring kernel filter must produce exactly one binary.
Missing SDKs and compiler failures are errors. Fingerprints include the codegen
schema, SDK dependency, command versions, and relevant Rust environment.

## PPy 0.3.1 frontend limitation

Analysis recognizes `Plugin.external_types()` and `DialectOperationSpec`, but
the canonical frontend does not connect them to arbitrary custom IR. A probe
with `fr.Tensor` parameters succeeds at analysis and yields no canonical module.
A scalar custom call also disappears; beside an ordinary scalar function, it
survives only as an empty declaration. A plugin pass cannot recover a function
discarded before shared passes.

The exact boundaries in PPy tag `v0.3.1` are:

- [Plugin.external_types and Plugin.call](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/plugins/base.py#L245)
  provide analysis types and operation notes, without an external-type-to-IR
  hook or an AST-to-IR callback.
- [eligible](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/backend/llvm/lowering.py#L230)
  requires native ABI types and rejects the tensor parameter.
- [Frontend.build](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/lowering/ast_to_ir.py#L254)
  applies that eligibility gate; `_drop` clears unsupported bodies.
- [_FunctionLowering._call](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/lowering/ast_to_ir.py#L1511)
  dispatches known calls without consuming generic plugin lowering notes.
- [canonical_ir_modules](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/driver/ir_pipeline.py#L155)
  skips modules without lowered functions before plugin passes run.

The minimal upstream fix is a public analyzed-type-to-IR mapping hook, canonical
signatures independent of CPU ABI eligibility, and frontend consumption of
declared custom operations with operands, attributes, effects, and source
locations. Explicit external compilation should diagnose rejected functions.
Regression tests must verify actual custom types and operations reach the
external backend. Reconstructing Python source inside this backend would bypass
the shared pipeline.

Separately, [Operation.spec_is_terminator](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/ir/model.py#L288)
uses the default registry even when verification receives a project registry.
The isolated public-registry compatibility measure described in
[Architecture](architecture.md) fixes recognition of `rngd.yield`; it does not
fix source lowering.

The same version's [IR build command](https://github.com/franknoh/PPy/blob/v0.3.1/src/ppy_compiler/driver/commands.py#L671)
calls the codec's `read(target)` without supplying the project's dialect
registry. Explicit backend activation therefore installs the same fixed metadata
in the public process registry for PPy 0.3.1. This lets the real `.ppyir` build
command decode the physical dialect before entering the backend.
