"""Project-scoped source semantics and RNGD lowering for PPy 0.3.2."""

from collections.abc import Sequence

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.ir import DialectRegistry, IRType, PassManager
from ppy_compiler.ir.transforms import PromoteSlots
from ppy_compiler.plugins.base import CallResult, Plugin

from .frontend import LowerBroadcast
from .physical import PhysicalDialect
from .semantic import (
    MUTABLE_TENSOR_NAME,
    TENSOR_NAME,
    SemanticDialect,
    SemanticTensor,
    recognize_call,
)
from .version import __version__


class FuriosaPlugin(Plugin):
    name = "furiosa"
    modules = ("ppy_furiosa",)
    api_version = 2

    def fingerprint(self) -> str:
        return f"api2:ppy-furiosa={__version__}:furiosa=1:rngd=1"

    def external_types(self) -> dict[str, str]:
        return {name: name for name in (TENSOR_NAME, MUTABLE_TENSOR_NAME)}

    def lower_type(self, type_: T.Type, facts: Facts) -> IRType | None:
        if not isinstance(type_, T.Instance) or type_.name not in {
            TENSOR_NAME,
            MUTABLE_TENSOR_NAME,
        }:
            return None
        try:
            return SemanticTensor.from_facts(type_, facts).to_ir()
        except ValueError:
            return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        return recognize_call(qualname, args, keywords)

    def register_dialects(self, registry: DialectRegistry) -> None:
        registry.register(PhysicalDialect())
        registry.register(SemanticDialect())

    def register_passes(self, manager: PassManager) -> None:
        manager.register_stage_pass("before-backend", PromoteSlots)
        manager.register_stage_pass("before-backend", LowerBroadcast)


def create_plugin(options: dict[str, object] | None = None) -> FuriosaPlugin:
    return FuriosaPlugin(options)
