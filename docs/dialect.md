# Physical dialect

Mappings are immutable, hashable expression trees encoded as PPy dialect types.
They support symbols, positive constants, division, modulo, constrained extent,
and padding. Ordered axes form a `Mapping`; Rust snippets are not accepted as
mapping data. Symbols must be valid nonreserved Rust identifiers.

```python
from ppy_furiosa.mapping import Mapping, Symbol

hidden = Symbol("H")
assert Mapping((hidden // 16, hidden % 16)).rust() == "m![H / 16, H % 16]"
```

`Tensor` records element format, memory tier, logical and local mappings, chip,
cluster, slice placement, and mutability. Format metadata covers bf16, f32,
f8e4m3, f4e2m1, and i32. Memory metadata names HBM, DM, TRF, and VRF; the
implemented broadcast uses bf16 HBM/DM tensors. TRF emission requires a lane
mapping and is explicitly unsupported by the current tensor emitter.

The implemented operation set is:

```text
rngd.to_dm
rngd.to_hbm
rngd.pipeline
    rngd.fetch
    rngd.switch
    rngd.collect
    rngd.commit_trim
    rngd.commit
    rngd.yield
```

The nested pipeline region carries a single SSA chain from its block argument
through the committed result. The supported switch is a 256-slice custom
broadcast. The verifier checks memory directions, matching transfer format and
logical shape, mutable HBM destinations, region structure, stage order, and
placement consistency. Backend validation rejects unsupported IR.

`broadcast_module(size=3840)` builds the supported example; size must be a
positive multiple of 16. The output logical shape is `[Copies, H]` with
`Copies=256`, while each slice stores `[H]`. This keeps replicated values
visible in the ABI instead of discarding them.
