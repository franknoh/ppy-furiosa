"""Physical tensor types; logical axes are distinct from physical placement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ppy_compiler.ir import DialectType, IRType

from .mapping import Const, Mapping


class Element(StrEnum):
    BF16 = "bf16"
    F32 = "f32"
    F8E4M3 = "f8e4m3"
    F4E2M1 = "f4e2m1"
    I32 = "i32"


class Memory(StrEnum):
    HBM = "hbm"
    DM = "dm"
    TRF = "trf"
    VRF = "vrf"


@dataclass(frozen=True, slots=True)
class Tensor:  # pylint: disable=too-many-instance-attributes
    element: Element
    memory: Memory
    logical: Mapping
    local: Mapping
    chip: Mapping = Mapping((Const(1),))
    cluster: Mapping = Mapping((Const(1).pad(2),))
    slices: Mapping = Mapping((Const(1).pad(256),))
    mutable: bool = False

    def to_ir(self) -> DialectType:
        return DialectType(
            "rngd",
            "tensor",
            (
                self.element.value,
                self.memory.value,
                self.logical.to_ir(),
                self.local.to_ir(),
                self.chip.to_ir(),
                self.cluster.to_ir(),
                self.slices.to_ir(),
                "mut" if self.mutable else "const",
            ),
        )

    @classmethod
    def from_ir(cls, value: IRType) -> Tensor:
        if not isinstance(value, DialectType) or (value.dialect, value.name) != ("rngd", "tensor"):
            raise ValueError("expected an RNGD tensor")
        if len(value.args) != 8:
            raise ValueError("RNGD tensor requires format, tier, five mappings and mutability")
        element, memory, *_, mutability = value.args
        if not isinstance(element, str) or not isinstance(memory, str):
            raise ValueError("tensor format and memory must be names")
        if mutability not in ("mut", "const"):
            raise ValueError("tensor mutability must be mut or const")
        mappings: list[Mapping] = []
        for mapping in value.args[2:7]:
            if not isinstance(mapping, IRType):
                raise ValueError("tensor layout must be a mapping")
            mappings.append(Mapping.from_ir(mapping))
        return cls(
            Element(element),
            Memory(memory),
            mappings[0],
            mappings[1],
            mappings[2],
            mappings[3],
            mappings[4],
            mutable=mutability == "mut",
        )

    def rust(self) -> str:
        args = [self.element.value, self.chip.rust()]
        if self.memory is Memory.HBM:
            args.append(self.logical.rust())
        else:
            args.extend((self.cluster.rust(), self.slices.rust()))
            if self.memory is Memory.TRF:
                raise ValueError("TRF requires an explicit register mapping; unsupported here")
            args.append(self.local.rust())
        name = {Memory.HBM: "HbmTensor", Memory.DM: "DmTensor", Memory.VRF: "VrfTensor"}
        return f"{name[self.memory]}<{', '.join(args)}>"
