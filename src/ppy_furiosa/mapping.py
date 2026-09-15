"""Immutable mapping expressions with a lossless PPy dialect-type encoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ppy_compiler.ir import DialectType, IRType

RUST_KEYWORDS = frozenset(
    "as break const continue crate else enum extern false fn for if impl in let loop match "
    "mod move mut pub ref return self Self static struct super trait true type unsafe use "
    "where while async await dyn abstract become box do final macro override priv typeof "
    "unsized virtual yield try gen".split()
)


def identifier(name: str) -> str:
    if not name or name in RUST_KEYWORDS or name == "_":
        raise ValueError(f"invalid Rust symbol {name!r}")
    if (
        not name.isascii()
        or not (name[0].isalpha() or name[0] == "_")
        or not all(char.isalnum() or char == "_" for char in name)
    ):
        raise ValueError(f"invalid Rust symbol {name!r}")
    return name


class ExprKind(StrEnum):
    SYMBOL = "symbol"
    CONST = "constant"
    DIV = "div"
    MOD = "mod"
    EXTENT = "extent"
    PAD = "pad"


@dataclass(frozen=True, slots=True)
class Expr:
    kind: ExprKind
    value: str | int
    base: Expr | None = None

    def __post_init__(self) -> None:
        if self.kind is ExprKind.SYMBOL:
            if not isinstance(self.value, str) or self.base is not None:
                raise ValueError("a symbol needs a name and no base")
            identifier(self.value)
        else:
            if not isinstance(self.value, int) or isinstance(self.value, bool) or self.value <= 0:
                raise ValueError("mapping dimensions and divisors must be positive integers")
            if (self.kind is ExprKind.CONST) != (self.base is None):
                raise ValueError("malformed mapping expression")
            if self.kind is ExprKind.PAD and self.base is not None:
                size = self.base.known_extent()
                if size is not None and self.value < size:
                    raise ValueError("padding cannot truncate the logical extent")

    def __floordiv__(self, divisor: int) -> Expr:
        return Expr(ExprKind.DIV, divisor, self)

    def __mod__(self, divisor: int) -> Expr:
        return Expr(ExprKind.MOD, divisor, self)

    def with_extent(self, extent: int) -> Expr:
        return Expr(ExprKind.EXTENT, extent, self)

    def pad(self, capacity: int) -> Expr:
        return Expr(ExprKind.PAD, capacity, self)

    def known_extent(self) -> int | None:
        if self.kind in {ExprKind.CONST, ExprKind.MOD, ExprKind.EXTENT, ExprKind.PAD}:
            assert isinstance(self.value, int)
            return self.value
        return None

    def rust(self) -> str:
        if self.base is None:
            return str(self.value)
        operator = {ExprKind.DIV: "/", ExprKind.MOD: "%", ExprKind.EXTENT: "=", ExprKind.PAD: "#"}[
            self.kind
        ]
        return f"{self.base.rust()} {operator} {self.value}"

    def to_ir(self) -> DialectType:
        args = (self.value,) if self.base is None else (self.base.to_ir(), self.value)
        return DialectType("rngd", self.kind.value, args)

    @classmethod
    def from_ir(cls, value: IRType) -> Expr:
        if not isinstance(value, DialectType) or value.dialect != "rngd":
            raise ValueError("expected an RNGD mapping expression")
        kind = ExprKind(value.name)
        if kind in {ExprKind.CONST, ExprKind.SYMBOL}:
            if len(value.args) != 1 or not isinstance(value.args[0], (str, int)):
                raise ValueError("malformed mapping leaf")
            return cls(kind, value.args[0])
        if len(value.args) != 2 or not isinstance(value.args[0], IRType):
            raise ValueError("malformed mapping expression")
        amount = value.args[1]
        if not isinstance(amount, int):
            raise ValueError("mapping extent must be an integer")
        return cls(kind, amount, cls.from_ir(value.args[0]))


def Symbol(name: str) -> Expr:  # pylint: disable=invalid-name
    return Expr(ExprKind.SYMBOL, name)


def Const(value: int) -> Expr:  # pylint: disable=invalid-name
    return Expr(ExprKind.CONST, value)


@dataclass(frozen=True, slots=True)
class Mapping:
    axes: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("a mapping needs at least one axis")

    def rust(self) -> str:
        return f"m![{', '.join(axis.rust() for axis in self.axes)}]"

    def to_ir(self) -> DialectType:
        return DialectType("rngd", "mapping", tuple(axis.to_ir() for axis in self.axes))

    @classmethod
    def from_ir(cls, value: IRType) -> Mapping:
        if not isinstance(value, DialectType) or (value.dialect, value.name) != ("rngd", "mapping"):
            raise ValueError("expected an RNGD mapping")
        axes: list[Expr] = []
        for axis in value.args:
            if not isinstance(axis, IRType):
                raise ValueError("mapping axes must be typed expressions")
            axes.append(Expr.from_ir(axis))
        return cls(tuple(axes))
