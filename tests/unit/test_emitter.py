"""Generated Rust is deterministic, source mapped, and rejects unsupported IR."""

from pathlib import Path

import pytest
from ppy_compiler.backend import BackendValidationError
from ppy_compiler.ir import Operation
from ppy_compiler.ir.codec import decode, encode

from ppy_furiosa.compatibility import ir_attributes
from ppy_furiosa.emitter import emit_rust
from ppy_furiosa.mapping import Const, Mapping, Symbol
from ppy_furiosa.physical import STREAM, broadcast_module, make_registry


@pytest.mark.parametrize("reserved", ["Context", "DmTensor", "HbmTensor", "SwitchConfig", "bf16"])
def test_axis_cannot_shadow_sdk_types(reserved: str) -> None:
    registry = make_registry()
    module = decode(encode(broadcast_module(), registry).replace("H", reserved), registry)
    with pytest.raises(BackendValidationError, match="reserved"):
        emit_rust(module)


def test_broadcast_golden() -> None:
    result = emit_rust(broadcast_module())
    assert result.source == (Path(__file__).parents[1] / "golden/broadcast.rs").read_text()
    assert result == emit_rust(broadcast_module())
    assert result.locations
    assert any(item.operation == "rngd.switch" for item in result.locations)
    for item in result.locations:
        assert item.source_file == "examples/broadcast/build_ir.py"
        assert 1 <= item.rust_line <= len(result.source.splitlines())


def test_rejects_unknown_operation() -> None:
    module = broadcast_module()
    next(module.functions["broadcast"].operations()).name = "rngd.secret"
    with pytest.raises(BackendValidationError, match="secret"):
        emit_rust(module)


def test_rejects_rust_identifier_injection() -> None:
    module = broadcast_module()
    module.attributes["rngd.axes"] = {"H]; panic!()": 3840}
    with pytest.raises(BackendValidationError, match="symbol"):
        emit_rust(module)


@pytest.mark.parametrize(
    "axes", [{"H": 3840}, {"H": 3839, "Copies": 256}, {"H": 3840, "Copies": 128}]
)
def test_rejects_invalid_broadcast_cardinality(axes: dict[str, int]) -> None:
    module = broadcast_module()
    ir_attributes(module)["rngd.axes"] = axes
    with pytest.raises(BackendValidationError, match="dimension|multiple of 16|256"):
        emit_rust(module)


def test_rejects_standalone_pipeline_stage() -> None:
    module = broadcast_module()
    entry = module.functions["broadcast"].body.blocks[0]
    entry.insert(
        0,
        Operation(
            "rngd.fetch",
            [entry.arguments[0]],
            [STREAM],
            {"time": Mapping((Const(1),)).to_ir(), "packet": Mapping((Symbol("H"),)).to_ir()},
        ),
    )
    with pytest.raises(BackendValidationError, match="pipeline|top-level"):
        emit_rust(module)


@pytest.mark.parametrize("operation_name", ["rngd.to_dm", "rngd.pipeline", "rngd.commit"])
@pytest.mark.parametrize("part", ["operands", "results"])
def test_short_operation_reports_backend_diagnostic(operation_name: str, part: str) -> None:
    module = broadcast_module()
    operation = next(
        op for op in module.functions["broadcast"].operations() if op.name == operation_name
    )
    getattr(operation, part).clear()
    with pytest.raises(BackendValidationError):
        emit_rust(module)
