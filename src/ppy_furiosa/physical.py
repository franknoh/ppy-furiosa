"""RNGD transfer and broadcast pipelines expressed in PPy's canonical SSA IR."""

from __future__ import annotations

from dataclasses import replace

from ppy_compiler.ir import (
    Builder,
    Dialect,
    DialectRegistry,
    DialectType,
    IRModule,
    Operation,
    OpSpec,
    SourceLocation,
    Value,
)
from ppy_compiler.ir.dialects.core import CoreDialect
from ppy_compiler.ir.verify import Checker

from .compatibility import (
    add_function,
    create_operation,
    enable_custom_terminators,
    ir_attributes,
)
from .mapping import Const, Expr, ExprKind, Mapping, Symbol
from .tensor import Element, Memory, Tensor

STREAM = DialectType("rngd", "stream")
PIPELINE_OPERATIONS = (
    "rngd.fetch",
    "rngd.switch",
    "rngd.collect",
    "rngd.commit_trim",
    "rngd.commit",
    "rngd.yield",
)


def _mapping(op: Operation, name: str) -> Mapping:
    value = ir_attributes(op).get(name)
    if not isinstance(value, DialectType):
        raise ValueError(f"{name} must be an RNGD mapping")
    return Mapping.from_ir(value)


def _verify_operation(op: Operation, checker: Checker) -> None:
    try:
        _check_operation(op)
    except (ValueError, IndexError) as error:
        checker.error(op, str(error))


def _check_operation(op: Operation) -> None:
    if op.name == "rngd.to_dm":
        source, target = Tensor.from_ir(op.operands[0].type), Tensor.from_ir(op.result.type)
        if source.memory not in {Memory.HBM, Memory.DM} or target.memory is not Memory.DM:
            raise ValueError("to_dm requires HBM/DM input and DM output")
        if source.logical != target.logical or source.element != target.element:
            raise ValueError("to_dm must preserve logical shape and format")
    elif op.name == "rngd.to_hbm":
        source, target = (Tensor.from_ir(value.type) for value in op.operands)
        if source.memory is not Memory.DM or target.memory is not Memory.HBM or not target.mutable:
            raise ValueError("to_hbm requires DM input and mutable HBM destination")
        if source.logical != target.logical or source.element != target.element:
            raise ValueError("to_hbm requires matching logical shape and format")
    elif op.name == "rngd.pipeline":
        _check_pipeline(op)
    elif op.name in {"rngd.fetch", "rngd.collect"}:
        _mapping(op, "time")
        _mapping(op, "packet")
    elif op.name == "rngd.switch":
        _mapping(op, "slices")
        _mapping(op, "packet")
        if ir_attributes(op).get("ring_size") != 256:
            raise ValueError("CustomBroadcast currently requires ring_size 256")
    elif op.name == "rngd.commit_trim":
        _mapping(op, "packet")


def _check_pipeline(op: Operation) -> None:
    if ir_attributes(op).get("context") not in ("main", "sub"):
        raise ValueError("pipeline context must be main or sub")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("pipeline requires one region with one block")
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1 or block.arguments[0].type != op.operands[0].type:
        raise ValueError("pipeline argument must match its input")
    if tuple(step.name for step in block.operations) != PIPELINE_OPERATIONS:
        raise ValueError(
            "broadcast pipeline requires fetch, switch, collect, commit_trim, commit, yield"
        )
    for step in block.operations:
        expected_results = 0 if step.name == "rngd.yield" else 1
        if len(step.operands) != 1 or len(step.results) != expected_results:
            raise ValueError(f"{step.name} has invalid operand/result cardinality")
    if block.operations[-2].result.type != op.result.type:
        raise ValueError("pipeline commit type must match its result")
    previous = block.arguments[0]
    for step in block.operations:
        if step.operands != [previous]:
            raise ValueError("pipeline stages must form one uninterrupted SSA chain")
        if step.results:
            previous = step.result
    if any(step.result.type != STREAM for step in block.operations[:4]):
        raise ValueError("fetch, switch, collect and commit_trim must produce RNGD streams")
    _check_pipeline_tensors(op)


def _check_pipeline_tensors(op: Operation) -> None:
    source = Tensor.from_ir(op.operands[0].type)
    target = Tensor.from_ir(op.result.type)
    if source.memory is not Memory.DM or target.memory is not Memory.DM:
        raise ValueError("broadcast pipeline input and result must be DM tensors")
    if (
        source.element != target.element
        or source.chip != target.chip
        or source.cluster != target.cluster
    ):
        raise ValueError("broadcast must preserve format, chip and cluster placement")
    _check_broadcast_layout(source, target)
    _check_broadcast_mappings(op, source, target)


def _check_broadcast_layout(source: Tensor, target: Tensor) -> None:
    if source.element is not Element.BF16:
        raise ValueError("broadcast currently supports BF16 only")
    if (
        len(source.logical.axes) != 1
        or source.logical.axes[0].kind is not ExprKind.SYMBOL
        or source.local != source.logical
    ):
        raise ValueError("broadcast input must contain one complete symbolic vector")
    if (
        source.chip != Mapping((Const(1),))
        or source.cluster != Mapping((Const(1).pad(2),))
        or source.slices != Mapping((Const(1).pad(256),))
        or source.mutable
    ):
        raise ValueError(
            "broadcast requires an immutable single-slice seed on one chip and cluster"
        )
    if (
        len(target.logical.axes) != 2
        or target.logical.axes[0].kind is not ExprKind.SYMBOL
        or target.logical.axes[0] == source.logical.axes[0]
        or target.logical.axes[1:] != source.logical.axes
    ):
        raise ValueError("broadcast result mapping must replicate the entire input across slices")
    expected = replace(source, logical=target.logical, slices=Mapping(target.logical.axes[:1]))
    if target != expected:
        raise ValueError("broadcast result mapping must replicate the entire input across slices")


def _check_broadcast_mappings(op: Operation, source: Tensor, target: Tensor) -> None:
    hidden = source.logical.axes[0]
    one, packet = Mapping((Const(1),)), Mapping((hidden % 16,))
    expected = (
        {"time": one, "packet": source.local},
        {"slices": target.slices, "packet": one},
        {"time": Mapping((hidden // 16,)), "packet": packet},
        {"packet": packet},
    )
    for step, mappings in zip(op.regions[0].blocks[0].operations[:4], expected, strict=True):
        for name, mapping in mappings.items():
            if _mapping(step, name) != mapping:
                raise ValueError(f"{step.name} {name} mapping is incompatible with BF16 broadcast")


class PhysicalDialect(Dialect):
    name = "rngd"

    def register_operations(self, registry: DialectRegistry) -> None:
        registry.add_op(OpSpec("rngd.to_dm", operands=1, results=1, verify=_verify_operation))
        registry.add_op(OpSpec("rngd.to_hbm", operands=2, results=0, verify=_verify_operation))
        registry.add_op(
            OpSpec(
                "rngd.pipeline",
                operands=1,
                results=1,
                regions=1,
                required_attributes=("context",),
                verify=_verify_operation,
            )
        )
        for name in ("fetch", "switch", "collect", "commit_trim", "commit"):
            registry.add_op(OpSpec(f"rngd.{name}", operands=1, results=1, verify=_verify_operation))
        registry.add_op(OpSpec("rngd.yield", operands=1, results=0, terminator=True))

    def verify_type(self, t: DialectType) -> str | None:
        try:
            if t.name == "tensor":
                Tensor.from_ir(t)
            elif t.name == "stream":
                if t.args:
                    raise ValueError("stream takes no arguments")
            elif t.name == "mapping":
                Mapping.from_ir(t)
            else:
                Expr.from_ir(t)
        except ValueError as error:
            return str(error)
        return None


def make_registry() -> DialectRegistry:
    enable_custom_terminators(PhysicalDialect())
    registry = DialectRegistry()
    registry.register(CoreDialect())
    registry.register(PhysicalDialect())
    return registry


def broadcast_module(size: int = 3840) -> IRModule:
    """A device kernel that writes 256 copies of a BF16 vector to HBM."""
    if size <= 0 or size % 16:
        raise ValueError("broadcast size must be a positive multiple of 16")
    hidden, copies = Symbol("H"), Symbol("Copies")
    vector, replicated = Mapping((hidden,)), Mapping((copies, hidden))
    source = Tensor(Element.BF16, Memory.HBM, vector, vector)
    destination = Tensor(Element.BF16, Memory.HBM, replicated, replicated, mutable=True)
    local = replace(source, memory=Memory.DM)
    result = replace(local, logical=replicated, slices=Mapping((copies,)))
    module = IRModule("broadcast", {"core": 1, "rngd": 1})
    ir_attributes(module)["rngd.axes"] = {"H": size, "Copies": 256}
    function = add_function(
        module,
        "broadcast",
        [("x", source.to_ir()), ("out", destination.to_ir())],
        [],
        values={"rngd.device": True},
    )
    entry = function.add_entry_block()
    builder = Builder(entry, SourceLocation("examples/broadcast/build_ir.py", 1))
    loaded = create_operation(builder, "rngd.to_dm", [entry.arguments[0]], [local.to_ir()]).result
    pipeline = _broadcast_pipeline(builder, loaded, local, result)
    create_operation(builder, "rngd.to_hbm", [pipeline.result, entry.arguments[1]])
    create_operation(builder, "core.ret")
    return module


def _broadcast_pipeline(
    builder: Builder, loaded: Value, local: Tensor, result: Tensor
) -> Operation:
    hidden, copies = Symbol("H"), Symbol("Copies")
    pipeline = create_operation(
        builder, "rngd.pipeline", [loaded], [result.to_ir()], {"context": "main"}
    )
    block = pipeline.add_region().add_block("body", [("input", local.to_ir())])
    inner = Builder(block, builder.location)
    one = Mapping.from_ir(local.chip.to_ir())
    value = create_operation(
        inner,
        "rngd.fetch",
        [block.arguments[0]],
        [STREAM],
        {"time": one.to_ir(), "packet": local.logical.to_ir()},
    ).result
    value = create_operation(
        inner,
        "rngd.switch",
        [value],
        [STREAM],
        {
            "slices": Mapping((copies,)).to_ir(),
            "packet": one.to_ir(),
            "ring_size": 256,
        },
    ).result
    value = create_operation(
        inner,
        "rngd.collect",
        [value],
        [STREAM],
        {
            "time": Mapping((hidden // 16,)).to_ir(),
            "packet": Mapping((hidden % 16,)).to_ir(),
        },
    ).result
    value = create_operation(
        inner, "rngd.commit_trim", [value], [STREAM], {"packet": Mapping((hidden % 16,)).to_ir()}
    ).result
    value = create_operation(inner, "rngd.commit", [value], [result.to_ir()]).result
    create_operation(inner, "rngd.yield", [value])
    return pipeline
