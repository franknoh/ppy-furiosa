"""Furiosa toolchain diagnostics and verified physical IR emission."""

import argparse
import sys
from pathlib import Path

from ppy_compiler.backend import BackendError
from ppy_compiler.ir.codec import CodecError, read

from .emitter import emit_rust
from .physical import make_registry
from .toolchain import FuriosaToolchain


def main() -> int:
    parser = argparse.ArgumentParser(prog="ppy-furiosa")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check Rust and Furiosa SDK availability")
    emit = commands.add_parser("emit-ir", help="emit Rust from canonical physical .ppyir")
    emit.add_argument("input", type=Path)
    emit.add_argument("-o", "--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "doctor":
            status = FuriosaToolchain().status()
            print(status.detail)
            return 0 if status.available else 1
        module = read(arguments.input, make_registry())
        arguments.output.write_text(emit_rust(module).source, encoding="utf-8")
    except (BackendError, CodecError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
