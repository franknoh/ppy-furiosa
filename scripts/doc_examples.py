"""Verify the README's emitted Rust and optionally build it through the PPy CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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


def check_build(root: Path, target: Path, output: Path, expected: str) -> None:
    """Compile through PPy's driver and require real, current SDK artifacts."""
    subprocess.run(
        [command("ppy"), "build", str(target), "--backend", "furiosa", "-o", str(output)],
        cwd=root,
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("generated_files"):
        raise RuntimeError("backend manifest contains no generated files")
    generated = [output / name for name in manifest["generated_files"]]
    binaries = [path for path in generated if path.suffix == ".bin"]
    schedules = [path for path in generated if path.name == "schedule.json"]
    rust_files = [path for path in generated if path.suffix == ".rs"]
    if len(binaries) != 1 or not binaries[0].stat().st_size or len(schedules) != 1:
        raise RuntimeError("backend build did not produce one device binary and schedule")
    if len(rust_files) != 1 or rust_files[0].read_text(encoding="utf-8") != expected:
        raise RuntimeError("build source differs from emitted Rust")
    print(f"SDK gate ({target.suffix}): 1 binary, 1 schedule; no hardware run")


def check_example(root: Path, directory: Path, *, build: bool) -> None:
    source = root / "examples/broadcast/kernel.ppy"
    ir_path = directory / "kernel.ppyir"
    rust_path = directory / "kernel.rs"
    subprocess.run(
        [command("ppy"), "emit", "furiosa-rust", str(source), "-o", str(rust_path)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [command("ppy"), "emit", "ir", str(source), "-o", str(ir_path)],
        cwd=root,
        check=True,
    )
    expected = (root / "tests/golden/source_broadcast.rs").read_text(encoding="utf-8")
    actual = rust_path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("source emit output differs from tests/golden/source_broadcast.rs")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if f"```python\n{source.read_text(encoding='utf-8')}```" not in readme:
        raise RuntimeError("README kernel.ppy example differs from the tested source")
    if f"```rust\n{actual}```" not in readme:
        raise RuntimeError("README Rust example differs from actual emitted output")
    if build:
        check_build(root, source, directory / "compiled-source", actual)
        check_build(root, ir_path, directory / "compiled-ir", actual)
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
