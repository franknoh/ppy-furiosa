"""Semantic contracts and source locations survive canonical lowering."""

import pytest
from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.ir import Builder, IRModule, PassContext, SourceLocation, verify

import ppy_furiosa
from ppy_furiosa.compatibility import add_function, create_operation
from ppy_furiosa.emitter import emit_rust
from ppy_furiosa.frontend import LowerBroadcast
from ppy_furiosa.physical import make_registry
from ppy_furiosa.plugin import FuriosaPlugin
from ppy_furiosa.semantic import SemanticTensor


def test_plugin_lowers_annotated_external_tensor() -> None:
    plugin = FuriosaPlugin()
    actual = plugin.lower_type(T.Instance("ppy_furiosa.Tensor"), Facts(shape=(32,), dtype="bf16"))
    assert actual == SemanticTensor((32,), "bf16", False).to_ir()
    assert plugin.lower_type(T.INT, Facts()) is None
    assert plugin.lower_type(T.Instance("ppy_furiosa.Tensor"), Facts()) is None


@pytest.mark.parametrize("shape,dtype", [((0,), "bf16"), ((32,), "f32"), ((), "bf16")])
def test_invalid_semantic_tensor(shape: tuple[int, ...], dtype: str) -> None:
    with pytest.raises(ValueError):
        SemanticTensor(shape, dtype, False)


def semantic_module() -> IRModule:
    module = IRModule("kernel", {"core": 1, "furiosa": 1})
    source = SemanticTensor((32,), "bf16", False)
    destination = SemanticTensor((256, 32), "bf16", True)
    function = add_function(
        module, "kernel", [("x", source.to_ir()), ("out", destination.to_ir())], []
    )
    block = function.add_entry_block()
    builder = Builder(block, SourceLocation("kernel.ppy", 6))
    value = create_operation(
        builder,
        "furiosa.broadcast",
        [block.arguments[0]],
        [source.broadcast(256).to_ir()],
        {"copies": 256},
    ).result
    builder.location = SourceLocation("kernel.ppy", 7)
    create_operation(builder, "furiosa.store", [block.arguments[1], value])
    create_operation(builder, "core.ret")
    return module


def test_semantic_operations_and_locations_are_lowered() -> None:
    registry = make_registry()
    FuriosaPlugin().register_dialects(registry)
    module = semantic_module()
    assert not verify(module, registry)
    assert LowerBroadcast().run(module, PassContext(registry))
    assert not verify(module, registry)
    assert module.dialects == {"core": 1, "rngd": 1}
    operations = list(module.functions["kernel"].operations())
    assert not any(op.name.startswith("furiosa.") for op in operations)
    for operation in operations:
        assert operation.location is not None
        if operation.name == "rngd.pipeline":
            assert operation.location.line == 6
        if operation.name == "rngd.to_hbm":
            assert operation.location.line == 7
    artifact = emit_rust(module)
    assert any(
        item.source_line == 6 and item.operation == "rngd.switch" for item in artifact.locations
    )


@pytest.mark.parametrize("name", ["furiosa.broadcast", "furiosa.store"])
def test_malformed_semantic_arity_is_diagnosed(name: str) -> None:
    registry = make_registry()
    FuriosaPlugin().register_dialects(registry)
    module = semantic_module()
    next(op for op in module.functions["kernel"].operations() if op.name == name).operands.clear()
    assert verify(module, registry)


def test_api_is_explicitly_compile_only() -> None:
    with pytest.raises(RuntimeError, match="compile-only"):
        ppy_furiosa.broadcast(ppy_furiosa.Tensor(), copies=256)
    with pytest.raises(RuntimeError, match="compile-only"):
        ppy_furiosa.store(ppy_furiosa.MutableTensor(), ppy_furiosa.Tensor())
