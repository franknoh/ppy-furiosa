"""Verify the README's emitted Rust and optionally build it through the PPy CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def command(name: str) -> str:
    """Require the installed executable selected by the active environment."""
    scripts = Path(sysconfig.get_path("scripts"))
    executable = scripts / (f"{name}.exe" if os.name == "nt" else name)
    if executable.is_file():
        return str(executable)
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"missing installed command {name!r}; run through uv run")
    return path


def check_example(root: Path, directory: Path, *, build: bool) -> None:
    ir_path = directory / "build/broadcast.ppyir"
    rust_path = directory / "build/broadcast.rs"
    subprocess.run(
        [sys.executable, str(root / "examples/broadcast/build_ir.py")],
        cwd=directory,
        check=True,
    )
    expected = (root / "tests/golden/broadcast.rs").read_text(encoding="utf-8")
    actual = rust_path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("example output differs from tests/golden/broadcast.rs")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if f"```rust\n{actual}```" not in readme:
        raise RuntimeError("README Rust example differs from actual emitted output")
    if build:
        output = directory / "compiled"
        subprocess.run(
            [command("ppy"), "build", str(ir_path), "--backend", "furiosa", "-o", str(output)],
            cwd=root,
            check=True,
        )
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("generated_files"):
            raise RuntimeError("backend manifest contains no generated files")
        generated = [output / name for name in manifest["generated_files"]]
        binaries = [path for path in generated if path.suffix == ".bin"]
        schedules = [path for path in generated if path.name == "schedule.json"]
        if not binaries or not all(path.stat().st_size for path in binaries) or not schedules:
            raise RuntimeError("backend build did not produce a device binary and schedule")
        print(f"SDK gate: {len(binaries)} binary, {len(schedules)} schedule; no hardware run")
    print("README example: emitted Rust matches the README and golden fixture")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="require a real Furiosa SDK build")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    temporary_root = root / "build" / "doc-examples"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=temporary_root) as directory:
        check_example(root, Path(directory), build=arguments.build)


if __name__ == "__main__":
    main()
