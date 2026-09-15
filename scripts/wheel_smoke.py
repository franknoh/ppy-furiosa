"""Smoke-test installed distribution discovery and physical IR outside source imports."""

import subprocess
import sys
import tempfile
from importlib.metadata import distribution, entry_points, version
from pathlib import Path

from ppy_compiler.driver.config import Config, PluginConfig
from ppy_compiler.ir.codec import decode, encode
from ppy_compiler.plugins.registry import load_plugins

from ppy_furiosa.emitter import emit_rust
from ppy_furiosa.physical import broadcast_module, make_registry


def check_source_emission(directory: Path) -> None:
    """Exercise the installed PPy command with the installed extension and no SDK."""
    (directory / "pyproject.toml").write_text(
        "[tool.ppy.plugins.furiosa]\nenabled = true\n"
        '[tool.ppy.backends.furiosa]\ncargo = "missing-furiosa-sdk"\n',
        encoding="utf-8",
    )
    fixture = Path(__file__).resolve().parents[1] / "examples/broadcast/kernel.ppy"
    (directory / "kernel.ppy").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    compiler = [sys.executable, "-m", "ppy_compiler", "emit", "furiosa-rust"]
    subprocess.run(
        [*compiler, "kernel.ppy", "-o", "kernel.rs"],
        cwd=directory,
        check=True,
    )
    if "CustomBroadcast" not in (directory / "kernel.rs").read_text(encoding="utf-8"):
        raise RuntimeError("installed source-to-Rust emission did not produce the broadcast")


def main() -> None:
    if any(ep.group == "console_scripts" for ep in distribution("ppy-furiosa").entry_points):
        raise RuntimeError("ppy-furiosa must integrate through PPy without a separate CLI")
    config = Config()
    config.plugins["furiosa"] = PluginConfig()
    plugins = load_plugins(config)
    if plugins.problems or plugins.for_module("ppy_furiosa") is None:
        raise RuntimeError(f"installed plugin failed: {plugins.problems}")
    backend = next(item for item in entry_points(group="ppy.backends") if item.name == "furiosa")
    if backend.load()({}).api_version != 1:
        raise RuntimeError("installed backend API differs from 1")
    module = decode(encode(broadcast_module(), make_registry()), make_registry())
    source = emit_rust(module).source
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "broadcast.rs"
        output.write_text(source, encoding="utf-8")
        if "CustomBroadcast" not in output.read_text(encoding="utf-8"):
            raise RuntimeError("installed emitter did not emit the broadcast pipeline")
        check_source_emission(Path(directory))
    print(f"Wheel smoke: ppy-furiosa {version('ppy-furiosa')}; PPy {version('ppy-lang')}")


if __name__ == "__main__":
    main()
