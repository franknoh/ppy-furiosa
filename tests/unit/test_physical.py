"""Physical IR verifies before it reaches Rust."""

from dataclasses import replace

import pytest
from ppy_compiler.ir import I32, parse_type, verify
from ppy_compiler.ir.codec import decode, encode

from ppy_furiosa.mapping import Mapping, Symbol
from ppy_furiosa.physical import broadcast_module, make_registry
from ppy_furiosa.tensor import Element, Memory, Tensor


@pytest.mark.parametrize("element", list(Element))
def test_tensor_roundtrip_preserves_layout_and_mutability(element: Element) -> None:
    shape = Mapping((Symbol("H"),))
    tensor = Tensor(element, Memory.HBM, shape, shape, mutable=True)
    assert Tensor.from_ir(parse_type(str(tensor.to_ir()))) == tensor


def test_broadcast_is_verified_and_losslessly_encoded() -> None:
    module = broadcast_module()
    registry = make_registry()
    assert not verify(module, registry)
    encoded = encode(module, registry)
    restored = decode(encoded, registry)
    assert not verify(restored, registry)
    assert encode(restored, registry) == encoded


def test_illegal_context_is_rejected() -> None:
    module = broadcast_module()
    pipeline = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.pipeline"
    )
    pipeline.attributes["context"] = "tdma"
    assert any("context" in str(error) for error in verify(module, make_registry()))


def test_write_requires_mutable_hbm() -> None:
    module = broadcast_module()
    store = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.to_hbm"
    )
    target = Tensor.from_ir(store.operands[1].type)
    store.operands[1].type = replace(target, mutable=False).to_ir()
    assert any("mutable HBM" in str(error) for error in verify(module, make_registry()))


def test_pipeline_must_commit_before_yield() -> None:
    module = broadcast_module()
    pipeline = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.pipeline"
    )
    block = pipeline.regions[0].blocks[0]
    block.operations[-2].name = "rngd.fetch"
    assert any("commit" in str(error) for error in verify(module, make_registry()))


@pytest.mark.parametrize("size", [0, 15, -1])
def test_invalid_broadcast_size(size: int) -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        broadcast_module(size)


@pytest.mark.parametrize(
    "stage,key",
    [
        ("fetch", "time"),
        ("fetch", "packet"),
        ("switch", "slices"),
        ("switch", "packet"),
        ("collect", "time"),
        ("collect", "packet"),
        ("commit_trim", "packet"),
    ],
)
def test_pipeline_rejects_incompatible_packet_mapping(stage: str, key: str) -> None:
    module = broadcast_module()
    operation = next(
        op for op in module.functions["broadcast"].operations() if op.name == f"rngd.{stage}"
    )
    operation.attributes[key] = Mapping((Symbol("H") % 8,)).to_ir()
    assert any("mapping" in str(error) for error in verify(module, make_registry()))


def test_pipeline_rejects_non_stream_intermediate() -> None:
    module = broadcast_module()
    operation = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.fetch"
    )
    operation.result.type = I32
    assert any("stream" in str(error) for error in verify(module, make_registry()))


@pytest.mark.parametrize("part", ["operands", "results"])
def test_malformed_short_stage_is_diagnosed(part: str) -> None:
    module = broadcast_module()
    operation = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.commit"
    )
    getattr(operation, part).clear()
    assert verify(module, make_registry())


def test_pipeline_rejects_unsupported_tensor_format() -> None:
    module = broadcast_module()
    pipeline = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.pipeline"
    )
    for value in (
        pipeline.operands[0],
        pipeline.result,
        pipeline.regions[0].blocks[0].operations[-2].result,
    ):
        value.type = replace(Tensor.from_ir(value.type), element=Element.F32).to_ir()
    pipeline.regions[0].blocks[0].arguments[0].type = pipeline.operands[0].type
    assert any("BF16" in str(error) for error in verify(module, make_registry()))


def test_pipeline_rejects_result_local_mapping_mismatch() -> None:
    module = broadcast_module()
    pipeline = next(
        op for op in module.functions["broadcast"].operations() if op.name == "rngd.pipeline"
    )
    changed = replace(
        Tensor.from_ir(pipeline.result.type), local=Mapping((Symbol("H") % 8,))
    ).to_ir()
    pipeline.result.type = changed
    pipeline.regions[0].blocks[0].operations[-2].result.type = changed
    assert any("result mapping" in str(error) for error in verify(module, make_registry()))
