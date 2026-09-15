"""Smoke-test installed distribution discovery and physical IR outside source imports."""

import tempfile
from importlib.metadata import entry_points, version
from pathlib import Path

from ppy_compiler.driver.config import Config, PluginConfig
from ppy_compiler.ir.codec import decode, encode
from ppy_compiler.plugins.registry import load_plugins

from ppy_furiosa.emitter import emit_rust
from ppy_furiosa.physical import broadcast_module, make_registry


def main() -> None:
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
    print(f"Wheel smoke: ppy-furiosa {version('ppy-furiosa')}; PPy {version('ppy-lang')}")


if __name__ == "__main__":
    main()
