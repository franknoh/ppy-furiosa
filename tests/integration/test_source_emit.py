"""Real installed PPy source emission through the Furiosa plugin and backend."""

import subprocess
import sys
from pathlib import Path

import pytest
from ppy_compiler.ir.codec import read

from ppy_furiosa.emitter import emit_rust
from ppy_furiosa.physical import make_registry


def source(size: int = 3840, dtype: str = "bf16", copies: int = 256, mutable: bool = True) -> str:
    output = f"ppy.Tensor[ppy.{dtype}, ({copies}, {size})]"
    if mutable:
        output = f"ppy.Mut[{output}]"
    return (
        "import ppy\nimport ppy_furiosa as fx\n\n"
        "def broadcast_kernel(\n"
        f"    x: ppy.Tensor[ppy.{dtype}, ({size},)],\n"
        f"    out: {output},\n"
        ") -> None:\n"
        f"    value = fx.broadcast(x, copies={copies})\n"
        "    fx.store(out, value)\n"
    )


def emit(
    tmp_path: Path, text: str, *, opt_level: int = 2, kind: str = "furiosa-rust"
) -> tuple[subprocess.CompletedProcess[str], Path]:
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.ppy]\nopt-level = {opt_level}\n[tool.ppy.plugins.furiosa]\nenabled = true\n",
        encoding="utf-8",
    )
    kernel = tmp_path / "kernel.ppy"
    kernel.write_text(text, encoding="utf-8")
    output = tmp_path / ("kernel.ppyir" if kind == "ir" else "kernel.rs")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "emit",
            kind,
            str(kernel),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output


@pytest.mark.parametrize("size", [32, 3840])
@pytest.mark.parametrize("opt_level", [0, 2])
def test_real_ppy_source_emit(tmp_path: Path, size: int, opt_level: int) -> None:
    result, output = emit(tmp_path, source(size), opt_level=opt_level)
    assert result.returncode == 0, result.stdout + result.stderr
    rust = output.read_text(encoding="utf-8")
    assert f"H = {size}" in rust
    assert "CustomBroadcast { ring_size: 256 }" in rust
    assert ".collect::<m![H / 16], m![H % 16]>()" in rust
    assert "to_hbm_view" in rust


@pytest.mark.parametrize(
    "text,operation,reason",
    [
        (source(31), "ppy_furiosa.broadcast", "multiple of 16"),
        (source(dtype="f32"), "ppy_furiosa.broadcast", "bf16 only"),
        (source(copies=128), "ppy_furiosa.broadcast", "copies=256"),
        (source(mutable=False), "ppy_furiosa.store", "ppy.Mut"),
        (source().replace("(256, 3840)", "(256, 32)"), "ppy_furiosa.store", "must match"),
        (
            source().replace("ppy.Tensor[ppy.bf16, (3840,)]", "ppy.Tensor"),
            "ppy_furiosa.broadcast",
            "ppy.Tensor[dtype, shape]",
        ),
    ],
)
def test_source_contract_errors_do_not_emit(
    tmp_path: Path, text: str, operation: str, reason: str
) -> None:
    result, output = emit(tmp_path, text)
    assert result.returncode != 0
    diagnostic = result.stdout + result.stderr
    assert "error[E1802]" in diagnostic and operation in diagnostic
    assert reason in diagnostic
    assert "Traceback" not in diagnostic
    assert not output.exists()


def test_borrowed_input_and_local_alias_preserve_tensor_facts(tmp_path: Path) -> None:
    text = source(32).replace(
        "x: ppy.Tensor[ppy.bf16, (32,)]", "x: ppy.Borrowed[ppy.Tensor[ppy.bf16, (32,)]]"
    )
    text = text.replace("    value =", "    destination = out\n    value =")
    text = text.replace("fx.store(out, value)", "fx.store(destination, value)")
    result, output = emit(tmp_path, text)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "to_hbm_view" in output.read_text(encoding="utf-8")


def test_unsupported_source_flow_is_a_diagnostic(tmp_path: Path) -> None:
    text = source() + "    fx.store(out, value)\n"
    result, output = emit(tmp_path, text)
    assert result.returncode != 0
    assert "unsupported source flow" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_source_ir_preserves_call_locations(tmp_path: Path) -> None:
    result, output = emit(tmp_path, source(32), opt_level=0, kind="ir")
    assert result.returncode == 0, result.stdout + result.stderr
    module = read(output, make_registry())
    artifact = emit_rust(module)
    locations = {(item.operation, item.source_line) for item in artifact.locations}
    assert ("rngd.switch", 8) in locations
    assert ("rngd.to_hbm", 9) in locations
    assert all(item.source_file == "kernel.ppy" for item in artifact.locations)


def test_checked_in_source_example_uses_real_plugin(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "example.rs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppy_compiler",
            "emit",
            "furiosa-rust",
            "examples/broadcast/kernel.ppy",
            "-o",
            str(output),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pub fn kernel_broadcast(" in output.read_text(encoding="utf-8")
