# ppy-furiosa

The current extension builds verified physical broadcast IR for Furiosa RNGD.
It registers through PPy's external plugin and backend mechanisms, emits Rust,
and invokes the Furiosa compiler for canonical `.ppyir` input.

The supported example reads a bf16 HBM vector and writes 256 copies to HBM.
Generated broadcast Rust has compiled with SDK 0.6.0. Hardware correctness and
the three MOA kernels are not implemented.

- [Architecture](architecture.md): component responsibilities.
- [Backend](backend.md): configuration, artifacts, and source compilation limit.
- [Physical dialect](dialect.md): supported types, mappings, and pipeline.
- [MOA baseline](moa2026.md): exact competition ABI and source audit.
- [Tuning status](tuning.md): current measurement boundary.
- [Development](development.md): reproducible checks.
