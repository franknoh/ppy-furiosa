"""Shape-checked broadcast semantics, independent of RNGD placement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect, EffectSet
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.ir import BF16, Dialect, DialectRegistry, DialectType, IRType, Operation, OpSpec
from ppy_compiler.ir.dialects import layout
from ppy_compiler.ir.dialects.tensor import describe
from ppy_compiler.ir.verify import Checker
from ppy_compiler.plugins.base import CallArgument, CallResult, DialectOperationSpec, RejectSpec

from .compatibility import ir_attributes


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
            "furiosa", "tensor", (BF16, "mut" if self.mutable else "const", *self.shape)
        )

    @classmethod
    def from_ir(cls, value: IRType, *, mutable: bool = False) -> SemanticTensor:
        common = describe(value)
        if common is not None:
            if common.dtype != BF16 or common.layout != layout.Layout():
                raise ValueError("Furiosa broadcast requires a row-major bf16 tensor")
            dimensions: list[int] = []
            for dimension in common.shape:
                if not isinstance(dimension, int) or isinstance(dimension, bool):
                    raise ValueError("Furiosa tensor dimensions must be static integers")
                dimensions.append(dimension)
            return cls(tuple(dimensions), "bf16", mutable)
        if not isinstance(value, DialectType) or (value.dialect, value.name) != (
            "furiosa",
            "tensor",
        ):
            raise ValueError("expected a semantic Furiosa tensor")
        if len(value.args) < 3:
            raise ValueError("Furiosa tensor requires dtype, mutability and shape")
        dtype, ownership, *sizes = value.args
        if dtype == BF16:
            dtype = "bf16"
        if not isinstance(dtype, str) or ownership not in ("mut", "const"):
            raise ValueError("invalid Furiosa tensor dtype or mutability")
        shape: list[int] = []
        for dimension in sizes:
            if not isinstance(dimension, int) or isinstance(dimension, bool):
                raise ValueError("Furiosa tensor dimensions must be static integers")
            shape.append(dimension)
        return cls(tuple(shape), dtype, ownership == "mut")

    @classmethod
    def from_facts(cls, type_: T.Type, facts: Facts) -> SemanticTensor:
        if not T.is_tensor(type_):
            raise ValueError("expected ppy.Tensor")
        if facts.shape is None or facts.dtype is None:
            raise ValueError("annotate tensors with ppy.Tensor[dtype, shape]")
        shape: list[int] = []
        for dimension in facts.shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool):
                raise ValueError("Furiosa tensor dimensions must be static integers")
            shape.append(dimension)
        dtype = "bf16" if facts.dtype == "bfloat16" else facts.dtype
        return cls(tuple(shape), dtype, facts.ownership == "mut")

    def broadcast(self, copies: object) -> SemanticTensor:
        if not isinstance(copies, int) or isinstance(copies, bool) or copies != 256:
            raise ValueError("Furiosa broadcast requires copies=256")
        if self.mutable or len(self.shape) != 1 or self.shape[0] % 16:
            raise ValueError("broadcast requires an immutable vector with size a multiple of 16")
        return SemanticTensor((copies, *self.shape), self.dtype, False)

    def check_store(self, value: SemanticTensor) -> None:
        if not self.mutable:
            raise ValueError("store destination must be annotated ppy.Mut[ppy.Tensor[...]]")
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
                T.TENSOR,
                Facts(shape=tensor.shape, dtype="bfloat16"),
                lowering=DialectOperationSpec(
                    "furiosa", "broadcast", keyword_attributes=("copies",)
                ),
                arguments=(CallArgument(0, "borrowed"),),
            )
        if len(args) != 2 or keywords:
            raise ValueError("store takes destination and value tensors")
        SemanticTensor.from_facts(*args[0]).check_store(SemanticTensor.from_facts(*args[1]))
        return CallResult(
            T.NONE,
            effects=EffectSet.of(Effect.WRITE_MEMORY),
            lowering=DialectOperationSpec("furiosa", "store"),
            arguments=(CallArgument(0, "mut"), CallArgument(1, "borrowed")),
        )
    except ValueError as error:
        return CallResult(T.NONE, lowering=RejectSpec(str(error)), reason=str(error))


def _verify_operation(op: Operation, checker: Checker) -> None:
    try:
        # Common IR stores ownership on function parameters, not the tensor type.
        # Source analysis checks the call contract; LowerBroadcast checks the
        # parameter ownership again before creating a writable physical tensor.
        value = SemanticTensor.from_ir(op.operands[0].type, mutable=op.name == "furiosa.store")
        if op.name == "furiosa.broadcast":
            expected = value.broadcast(ir_attributes(op).get("copies"))
            if SemanticTensor.from_ir(op.result.type) != expected:
                raise ValueError("broadcast result shape, dtype and mutability must match")
        else:
            value.check_store(SemanticTensor.from_ir(op.operands[1].type))
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
