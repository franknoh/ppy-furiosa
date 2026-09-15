"""Project-scoped registration of the physical RNGD dialect."""

from ppy_compiler.ir import DialectRegistry
from ppy_compiler.plugins.base import Plugin

from .physical import PhysicalDialect, make_registry
from .version import __version__


class FuriosaPlugin(Plugin):
    name = "furiosa"
    modules = ("ppy_furiosa",)
    api_version = 2

    def fingerprint(self) -> str:
        return f"api2:ppy-furiosa={__version__}:rngd=1"

    def register_dialects(self, registry: DialectRegistry) -> None:
        make_registry()
        registry.register(PhysicalDialect())


def create_plugin(options: dict[str, object] | None = None) -> FuriosaPlugin:
    return FuriosaPlugin(options)
