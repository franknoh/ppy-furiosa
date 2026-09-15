"""Deterministic syntax-only emission of verified physical RNGD IR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from ppy_compiler.backend import BackendValidationError
from ppy_compiler.ir import DialectType, IRFunction, IRModule, Operation, Value, verify

from .compatibility import ir_attributes
from .mapping import Mapping, identifier
from .physical import PIPELINE_OPERATIONS, make_registry
from .tensor import Element, Memory, Tensor


@dataclass(frozen=True, slots=True)
class SourceMapEntry:
    rust_line: int
    source_file: str
    source_line: int
    operation: str


@dataclass(frozen=True, slots=True)
class RustArtifact:
    source: str
    locations: tuple[SourceMapEntry, ...]


def validate(module: IRModule) -> None:
    errors = verify(module, make_registry())
    if errors:
        raise BackendValidationError("furiosa", "; ".join(map(str, errors)))
    if not module.functions or module.globals:
        raise BackendValidationError("furiosa", "empty module or module globals")
    try:
        axes = dict(_axes(module))
        for function in module.functions.values():
            identifier(function.name)
            if function.is_declaration or len(function.body.blocks) != 1 or function.results:
                raise ValueError(
                    f"{function.name} requires a defined, single-block void device function"
                )
            if ir_attributes(function).get("rngd.device") is not True:
                raise ValueError(f"{function.name} is not an RNGD device function")
            for name, type_ in function.params:
                identifier(name)
                if name == "ctx" or name.startswith("v") and name[1:].isdigit():
                    raise ValueError(f"reserved generated parameter name {name!r}")
                Tensor.from_ir(type_).rust()
            for op in function.operations():
                if op.name not in {
                    "rngd.to_dm",
                    "rngd.to_hbm",
                    "rngd.pipeline",
                    *PIPELINE_OPERATIONS,
                    "core.ret",
                }:
                    raise BackendValidationError("furiosa", op.name, location=op.location)
            _validate_broadcast_function(function, axes)
    except ValueError as error:
        raise BackendValidationError("furiosa", str(error)) from error


def _validate_broadcast_function(function: IRFunction, axes: dict[str, int]) -> None:
    entry = function.body.blocks[0]
    if tuple(op.name for op in entry.operations) != (
        "rngd.to_dm",
        "rngd.pipeline",
        "rngd.to_hbm",
        "core.ret",
    ):
        raise ValueError("supported top-level body requires to_dm, pipeline, to_hbm, ret")
    if len(entry.arguments) != 2:
        raise ValueError("broadcast requires input and output tensor parameters")
    load, pipeline, store, _ = entry.operations
    if (
        load.operands != [entry.arguments[0]]
        or pipeline.operands != [load.result]
        or store.operands != [pipeline.result, entry.arguments[1]]
    ):
        raise ValueError("broadcast transfers must connect the input, pipeline and output")
    source, target = (Tensor.from_ir(argument.type) for argument in entry.arguments)
    local, result = Tensor.from_ir(load.result.type), Tensor.from_ir(pipeline.result.type)
    if source != replace(local, memory=Memory.HBM):
        raise ValueError("broadcast input transfer must preserve all tensor mappings")
    expected = Tensor(
        Element.BF16, Memory.HBM, result.logical, result.logical, chip=result.chip, mutable=True
    )
    if target != expected:
        raise ValueError(
            "broadcast output requires complete HBM mapping with preserved chip placement"
        )
    hidden, copies = local.logical.axes[0].value, result.logical.axes[0].value
    if not isinstance(hidden, str) or not isinstance(copies, str):
        raise ValueError("broadcast dimensions must be declared symbols")
    if hidden not in axes or copies not in axes:
        raise ValueError("every broadcast dimension must be declared in rngd.axes")
    if axes[hidden] % 16:
        raise ValueError("BF16 broadcast vector extent must be a multiple of 16")
    if axes[copies] != 256:
        raise ValueError("CustomBroadcast requires 256 destination slices")


def _axes(module: IRModule) -> tuple[tuple[str, int], ...]:
    value = ir_attributes(module).get("rngd.axes")
    if not isinstance(value, dict):
        raise ValueError("rngd.axes must declare each symbolic dimension")
    axes: list[tuple[str, int]] = []
    for name, extent in cast(dict[object, object], value).items():
        if (
            not isinstance(name, str)
            or not isinstance(extent, int)
            or extent.__class__ is not int
            or extent <= 0
        ):
            raise ValueError("axis names need positive integer extents")
        if name in {
            "Context",
            "DmTensor",
            "HbmTensor",
            "VrfTensor",
            "TrfTensor",
            "SwitchConfig",
            "bf16",
            "f32",
            "f64",
            "i32",
            "f8e4m3",
            "f4e2m1",
        }:
            raise ValueError(f"axis symbol {name!r} is reserved by the Furiosa SDK")
        axes.append((identifier(name), extent))
    return tuple(sorted(axes))


def _mapping(op: Operation, key: str) -> str:
    value = ir_attributes(op)[key]
    assert isinstance(value, DialectType)
    return Mapping.from_ir(value).rust()


def _pipeline_step(step: Operation) -> str | None:
    if step.name in {"rngd.fetch", "rngd.collect"}:
        return (
            f"        .{step.local_name}::<{_mapping(step, 'time')}, {_mapping(step, 'packet')}>()"
        )
    if step.name == "rngd.switch":
        return (
            f"        .switch::<{_mapping(step, 'slices')}, {_mapping(step, 'packet')}>"
            f"(SwitchConfig::CustomBroadcast {{ ring_size: {ir_attributes(step)['ring_size']} }})"
        )
    if step.name == "rngd.commit_trim":
        return f"        .commit_trim::<{_mapping(step, 'packet')}>()"
    if step.name == "rngd.commit":
        return "        .commit();"
    return None


def emit_rust(module: IRModule) -> RustArtifact:
    validate(module)
    lines = [
        "#![feature(register_tool)]",
        "#![register_tool(furiosa_opt)]",
        "",
        "use furiosa_opt_std::prelude::*;",
        "",
        "axes![" + ", ".join(f"{name} = {size}" for name, size in _axes(module)) + "];",
        "",
    ]
    locations: list[SourceMapEntry] = []

    def append(line: str, op: Operation) -> None:
        lines.append(line)
        if op.location:
            locations.append(
                SourceMapEntry(len(lines), op.location.file, op.location.line, op.name)
            )

    for function in module.functions.values():
        lines.extend(["#[device(chip = 1)]", f"pub fn {function.name}(", "    ctx: &mut Context,"])
        for name, type_ in function.params:
            tensor = Tensor.from_ir(type_)
            lines.append(f"    {name}: &{'mut ' if tensor.mutable else ''}{tensor.rust()},")
        lines.append(") {")
        entry = function.body.blocks[0]
        names: dict[Value, str] = {
            value: name for value, (name, _) in zip(entry.arguments, function.params, strict=True)
        }
        for op in entry.operations:
            if op.results:
                names[op.result] = f"v{len(names) - len(entry.arguments)}"
            if op.name == "rngd.to_dm":
                target = Tensor.from_ir(op.result.type).rust()
                append(
                    f"    let {names[op.result]}: {target} = "
                    f"{names[op.operands[0]]}.to_dm(&mut ctx.tdma);",
                    op,
                )
            elif op.name == "rngd.to_hbm":
                source, target = (names[value] for value in op.operands)
                append(f"    {source}.view().to_hbm_view(&mut ctx.tdma, {target}.view_mut());", op)
            elif op.name == "rngd.pipeline":
                target = Tensor.from_ir(op.result.type).rust()
                append(
                    f"    let {names[op.result]}: {target} = ctx.{ir_attributes(op)['context']}",
                    op,
                )
                append(f"        .begin({names[op.operands[0]]}.view())", op)
                for step in op.regions[0].blocks[0].operations:
                    line = _pipeline_step(step)
                    if line is not None:
                        append(line, step)
            elif op.name == "core.ret":
                append("}", op)
        lines.append("")
    return RustArtifact("\n".join(lines), tuple(locations))
