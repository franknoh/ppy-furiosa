"""Narrow public-API adapters for PPy 0.3.1 metadata and typing defects.

PPy's Attribute alias contains unparameterized tuple/dict members. Treat
attribute payloads as objects here; the dialect validates their actual shapes.
The protocols change only static typing and forward to the original public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from typing import Protocol, cast

from ppy_compiler.ir import (
    Builder,
    Dialect,
    IRFunction,
    IRModule,
    IRType,
    Operation,
    Value,
    registry,
)


class _Attributed(Protocol):
    attributes: dict[str, object]


class _Builder(Protocol):
    def create(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[IRType] = (),
        attributes: dict[str, object] | None = None,
    ) -> Operation: ...


class _Module(Protocol):
    def add_function(
        self,
        name: str,
        params: Sequence[tuple[str, IRType]],
        results: Sequence[IRType],
        *,
        attributes: dict[str, object] | None = None,
    ) -> IRFunction: ...


def ir_attributes(owner: IRModule | IRFunction | Operation) -> dict[str, object]:
    """Expose untyped upstream payloads for explicit runtime validation."""
    return cast(_Attributed, owner).attributes


def create_operation(
    builder: Builder,
    name: str,
    operands: Sequence[Value] = (),
    result_types: Sequence[IRType] = (),
    values: dict[str, object] | None = None,
) -> Operation:
    return cast(_Builder, builder).create(name, operands, result_types, values)


def add_function(
    module: IRModule,
    name: str,
    params: Sequence[tuple[str, IRType]],
    results: Sequence[IRType],
    values: dict[str, object] | None = None,
) -> IRFunction:
    return cast(_Module, module).add_function(name, params, results, attributes=values)


def enable_custom_terminators(dialect: Dialect) -> None:
    """Register fixed metadata when the integration is explicitly enabled.

    PPy 0.3.1's verify._verify_block_structure calls Operation.spec_is_terminator,
    which consults the process registry instead of the supplied project registry.
    Block.terminator/successors share the defect. This public registration fixes
    those lookups without monkeypatching; normal project registration is retained.
    The driver also reads .ppyir through the process registry. Recheck both
    boundaries when upgrading PPy; this workaround applies only to 0.3.1.
    """
    if version("ppy-lang") == "0.3.1":
        registry().register(dialect)
