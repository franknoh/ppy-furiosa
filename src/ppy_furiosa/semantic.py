"""Shape-checked broadcast semantics, independent of RNGD placement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect, EffectSet
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.ir import Dialect, DialectRegistry, DialectType, IRType, Operation, OpSpec
from ppy_compiler.ir.verify import Checker
from ppy_compiler.plugins.base import CallResult, DialectOperationSpec, RejectSpec

from .compatibility import ir_attributes

TENSOR_NAME = "ppy_furiosa.Tensor"
MUTABLE_TENSOR_NAME = "ppy_furiosa.MutableTensor"


@dataclass(frozen=True, slots=True)
class SemanticTensor:
    shape: tuple[int, ...]
    dtype: str
    mutable: bool

    def __post_init__(self) -> None:
        if not self.shape or any(size.__class__ is not int or size <= 0 for size in self.shape):
            raise ValueError("Furiosa tensor requires a positive, static shape")
        if self.dtype != "bf16":
            raise ValueError("Furiosa broadcast currently supports bf16 only")

    def to_ir(self) -> DialectType:
        return DialectType(
            "furiosa", "tensor", (self.dtype, "mut" if self.mutable else "const", *self.shape)
        )

    @classmethod
    def from_ir(cls, value: IRType) -> SemanticTensor:
        if not isinstance(value, DialectType) or (value.dialect, value.name) != (
            "furiosa",
            "tensor",
        ):
            raise ValueError("expected a semantic Furiosa tensor")
        if len(value.args) < 3:
            raise ValueError("Furiosa tensor requires dtype, mutability and shape")
        dtype, ownership, *dimensions = value.args
        if not isinstance(dtype, str) or ownership not in ("mut", "const"):
            raise ValueError("invalid Furiosa tensor dtype or mutability")
        shape: list[int] = []
        for dimension in dimensions:
            if not isinstance(dimension, int) or isinstance(dimension, bool):
                raise ValueError("Furiosa tensor dimensions must be static integers")
            shape.append(dimension)
        return cls(tuple(shape), dtype, ownership == "mut")

    @classmethod
    def from_facts(cls, type_: T.Type, facts: Facts) -> SemanticTensor:
        if not isinstance(type_, T.Instance) or type_.name not in {
            TENSOR_NAME,
            MUTABLE_TENSOR_NAME,
        }:
            raise ValueError("expected ppy_furiosa.Tensor")
        if facts.shape is None or facts.dtype is None:
            raise ValueError("annotate Furiosa tensors with ppy.Shape and ppy.DType")
        shape: list[int] = []
        for dimension in facts.shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool):
                raise ValueError("Furiosa tensor dimensions must be static integers")
            shape.append(dimension)
        return cls(
            tuple(shape), facts.dtype, type_.name == MUTABLE_TENSOR_NAME or facts.ownership == "mut"
        )

    def broadcast(self, copies: object) -> SemanticTensor:
        if not isinstance(copies, int) or isinstance(copies, bool) or copies != 256:
            raise ValueError("Furiosa broadcast requires copies=256")
        if self.mutable or len(self.shape) != 1 or self.shape[0] % 16:
            raise ValueError("broadcast requires an immutable vector with size a multiple of 16")
        return SemanticTensor((copies, *self.shape), self.dtype, False)

    def check_store(self, value: SemanticTensor) -> None:
        if not self.mutable:
            raise ValueError("store destination must be annotated ppy_furiosa.MutableTensor")
        if (self.shape, self.dtype) != (value.shape, value.dtype):
            raise ValueError("store value shape and dtype must match the destination")


def recognize_call(
    qualname: str,
    args: Sequence[tuple[T.Type, Facts]],
    keywords: dict[str, tuple[T.Type, Facts]],
) -> CallResult | None:
    if qualname not in {"ppy_furiosa.broadcast", "ppy_furiosa.store"}:
        return None
    try:
        if qualname == "ppy_furiosa.broadcast":
            if len(args) != 1 or set(keywords) != {"copies"}:
                raise ValueError("broadcast takes one tensor and keyword copies=256")
            facts = keywords["copies"][1]
            if not facts.has_constant:
                raise ValueError("broadcast copies must be a compile-time constant")
            tensor = SemanticTensor.from_facts(*args[0]).broadcast(facts.constant)
            return CallResult(
                args[0][0],
                Facts(shape=tensor.shape, dtype=tensor.dtype),
                lowering=DialectOperationSpec(
                    "furiosa", "broadcast", keyword_attributes=("copies",)
                ),
            )
        if len(args) != 2 or keywords:
            raise ValueError("store takes destination and value tensors")
        SemanticTensor.from_facts(*args[0]).check_store(SemanticTensor.from_facts(*args[1]))
        return CallResult(
            T.NONE,
            effects=EffectSet.of(Effect.WRITE_MEMORY),
            lowering=DialectOperationSpec("furiosa", "store"),
        )
    except ValueError as error:
        return CallResult(T.NONE, lowering=RejectSpec(str(error)), reason=str(error))


def _verify_operation(op: Operation, checker: Checker) -> None:
    try:
        tensor = SemanticTensor.from_ir(op.operands[0].type)
        if op.name == "furiosa.broadcast":
            expected = tensor.broadcast(ir_attributes(op).get("copies"))
            if SemanticTensor.from_ir(op.result.type) != expected:
                raise ValueError("broadcast result shape, dtype and mutability must match")
        else:
            tensor.check_store(SemanticTensor.from_ir(op.operands[1].type))
    except ValueError as error:
        checker.error(op, str(error))


class SemanticDialect(Dialect):
    name = "furiosa"

    def register_operations(self, registry: DialectRegistry) -> None:
        registry.add_op(
            OpSpec(
                "furiosa.broadcast",
                operands=1,
                results=1,
                pure=True,
                required_attributes=("copies",),
                verify=_verify_operation,
            )
        )
        registry.add_op(OpSpec("furiosa.store", operands=2, results=0, verify=_verify_operation))

    def verify_type(self, t: DialectType) -> str | None:
        try:
            SemanticTensor.from_ir(t)
        except ValueError as error:
            return str(error)
        return None
