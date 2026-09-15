# Architecture

`plugin.py` registers the `rngd` physical dialect with PPy. The plugin does not
generate Rust. `physical.py` constructs and verifies the supported broadcast
pipeline using PPy's canonical SSA operations and a nested region.

`mapping.py` stores immutable mapping expressions. `tensor.py` preserves element
format, memory tier, logical/local mappings, placement, and mutability.
`emitter.py` validates physical IR and returns deterministic Rust with generated
line locations. `backend.py` implements PPy backend API 1, owns output crates,
and delegates SDK execution to `toolchain.py`.

The implemented path is:

```text
physical Python builder -> canonical .ppyir -> PPy external backend
    -> validated physical IR -> Rust crate -> furiosa-opt -> binary + schedule
```

The backend consumes canonical IR. There is no backend AST parser, semantic
kernel shortcut, or MOA-specific fused operation. Generic semantic lowering is
blocked by the frontend connection described in [Backend](backend.md).

PPy 0.3.1 also resolves custom terminators through its process registry in some
IR queries and reads `.ppyir` without passing the project's dialect registry.
`compatibility.py` enables deterministic dialect metadata through
the public registry API when integration is explicitly loaded. Ordinary project
registration remains active; package import does not install the metadata.
