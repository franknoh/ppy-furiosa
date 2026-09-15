"""Installed entry points are exercised through the real PPy registries."""

import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

from ppy_compiler.driver.config import Config, PluginConfig
from ppy_compiler.ir.codec import write
from ppy_compiler.plugins.registry import load_plugins

from ppy_furiosa.physical import broadcast_module, make_registry


def test_plugin_discovery_loads_dialect() -> None:
    config = Config()
    config.plugins["furiosa"] = PluginConfig()
    registry = load_plugins(config)
    assert not registry.problems
    assert registry.for_module("ppy_furiosa") is not None
    assert registry.dialect_registry().op_spec("rngd.pipeline") is not None


def test_backend_entry_point_and_format_ownership() -> None:
    backend = next(ep for ep in entry_points(group="ppy.backends") if ep.name == "furiosa")
    assert backend.load()({}).api_version == 1
    format_entry = next(
        ep for ep in entry_points(group="ppy.backend-formats") if ep.name == "furiosa-rust"
    )
    assert format_entry.value == "furiosa"


def test_cli_emits_saved_physical_ir(tmp_path: Path) -> None:
    path = tmp_path / "broadcast.ppyir"
    output = tmp_path / "broadcast.rs"
    write(broadcast_module(), path, make_registry())
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ppy_furiosa.cli import main; raise SystemExit(main())",
            "emit-ir",
            str(path),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "CustomBroadcast" in output.read_text()


def test_ppy_build_reports_missing_sdk_without_creating_artifacts(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy.plugins.furiosa]\nenabled = true\n"
        '[tool.ppy.backends.furiosa]\ncargo = "missing-furiosa-cargo"\n',
        encoding="utf-8",
    )
    source, output = tmp_path / "broadcast.ppyir", tmp_path / "output"
    write(broadcast_module(), source, make_registry())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "build",
            str(source),
            "--backend",
            "furiosa",
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "missing-furiosa-cargo" in result.stderr + result.stdout
    assert not output.exists()
