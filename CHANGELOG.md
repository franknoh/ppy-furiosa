# Changelog

## Unreleased

- Register the external PPy plugin, backend API 1, and `furiosa-rust` format.
- Represent immutable RNGD mappings and tensor placement in canonical PPy IR.
- Verify and emit an HBM-to-DM broadcast pipeline with an HBM destination.
- Build canonical `.ppyir` through the PPy backend into a Rust crate, device
  artifact, source map, schedule, and manifest.
- Report missing Furiosa tools without requiring them at package import.
- Document the PPy 0.3.1 frontend extension gap and official MOA baseline.

Semantic source compilation, MOA kernel generation, hardware correctness, and
autotuning remain unavailable.
