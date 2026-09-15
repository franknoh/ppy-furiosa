"""Static typing adapters for PPy's public IR attribute APIs.

PPy's Attribute alias contains unparameterized tuple/dict members. Treat
attribute payloads as objects here; the dialect validates their actual shapes.
The protocols change only static typing and forward to the original public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from ppy_compiler.ir import (
    Builder,
    IRFunction,
    IRModule,
    IRType,
    Operation,
    Value,
)


class _Attributed(Protocol):
    attributes: dict[str, object]


class _Function(Protocol):
    param_attributes: list[dict[str, object]]


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


def parameter_attributes(function: IRFunction) -> list[dict[str, object]]:
    """Expose source ownership metadata through the public function API."""
    return cast(_Function, function).param_attributes


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
