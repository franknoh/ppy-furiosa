"""Write the BF16 broadcast example as canonical PPy IR and emitted Rust."""

from pathlib import Path

from ppy_compiler.ir.codec import write

from ppy_furiosa.emitter import emit_rust
from ppy_furiosa.physical import broadcast_module, make_registry

if __name__ == "__main__":
    output = Path("build/broadcast.ppyir")
    output.parent.mkdir(parents=True, exist_ok=True)
    module = broadcast_module()
    write(module, output, make_registry())
    output.with_suffix(".rs").write_text(emit_rust(module).source, encoding="utf-8")
    print(output)
    print(output.with_suffix(".rs"))
