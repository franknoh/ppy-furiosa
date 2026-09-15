"""Exercise installed CLI emission and optionally compile the physical example."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

from ppy_compiler.ir.codec import write

from ppy_furiosa.physical import broadcast_module, make_registry


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
    ir_path = directory / "broadcast.ppyir"
    rust_path = directory / "broadcast.rs"
    write(broadcast_module(), ir_path, make_registry())
    subprocess.run(
        [command("ppy-furiosa"), "emit-ir", str(ir_path), "-o", str(rust_path)],
        cwd=root,
        check=True,
    )
    expected = (root / "tests/golden/broadcast.rs").read_text(encoding="utf-8")
    actual = rust_path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("documented CLI output differs from tests/golden/broadcast.rs")
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
    print("Documentation example: canonical IR and installed CLI golden emission passed")


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
