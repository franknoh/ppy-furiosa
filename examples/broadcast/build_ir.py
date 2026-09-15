"""Write the verified BF16 broadcast example as canonical PPy IR."""

from pathlib import Path

from ppy_compiler.ir.codec import write

from ppy_furiosa.physical import broadcast_module, make_registry

if __name__ == "__main__":
    output = Path("build/broadcast.ppyir")
    output.parent.mkdir(parents=True, exist_ok=True)
    write(broadcast_module(), output, make_registry())
    print(output)
