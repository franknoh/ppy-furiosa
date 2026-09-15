"""PPy backend interface 1: physical IR to Rust crates and device artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from ppy_compiler.backend import (
    Backend,
    BackendContext,
    BackendError,
    BuildResult,
    EmitFormat,
    ToolchainStatus,
)
from ppy_compiler.ir import IRModule

from .emitter import emit_rust, validate
from .mapping import identifier
from .toolchain import FuriosaToolchain
from .version import __version__


class FuriosaBackend(Backend):
    name = "furiosa"
    api_version = 1

    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        super().__init__(options)
        unknown = self.options.keys() - {"target", "cargo", "rustc"}
        if unknown:
            raise BackendError(f"unknown Furiosa options: {', '.join(sorted(unknown))}")
        for key, value in self.options.items():
            if not isinstance(value, str) or not value.strip():
                raise BackendError(f"Furiosa option {key} must be a nonempty string")
        if self.options.get("target", "rngd") != "rngd":
            raise BackendError("Furiosa target must be rngd")
        self.toolchain = FuriosaToolchain(
            str(self.options.get("cargo", "cargo")), str(self.options.get("rustc", "rustc"))
        )

    def fingerprint(self) -> str:
        return f"rngd-rust-1:std=0.6.0:{self.toolchain.fingerprint()}"

    def emit_formats(self) -> tuple[EmitFormat, ...]:
        return (
            EmitFormat(
                "furiosa-rust",
                ".rs",
                description="Furiosa furiosa-opt Rust DSL source",
                requires_toolchain=False,
            ),
        )

    def validate(self, module: IRModule, context: BackendContext) -> None:
        if context.target not in ("", "rngd"):
            raise BackendError(f"unsupported Furiosa target {context.target!r}")
        validate(module)

    def emit(  # pylint: disable=redefined-builtin
        self, module: IRModule, format: str, context: BackendContext
    ) -> str:
        if format != "furiosa-rust":
            raise BackendError(f"unsupported Furiosa emit format {format!r}")
        self.validate(module, context)
        return emit_rust(module).source

    def toolchain_status(self) -> ToolchainStatus:
        return self.toolchain.status()

    def build(
        self, modules: Mapping[str, IRModule], output: Path, context: BackendContext
    ) -> BuildResult:
        output = output.resolve()
        if not modules:
            raise BackendError("cannot build empty Furiosa program")
        for module in modules.values():
            self.validate(module, context)
        manifest = output / "manifest.json"
        # Once any source changes, the previous build identity is no longer valid.
        manifest.unlink(missing_ok=True)
        outputs: list[Path] = []
        for name, module in sorted(modules.items()):
            # A qualified PPy name is encoded, never used as an unchecked path.
            component = name.replace("_", "_u").replace(".", "_d")
            try:
                identifier(component)
            except ValueError as error:
                raise BackendError(str(error)) from error
            crate = output / component
            outputs.extend(_write_crate(crate, component, module))
            for function in module.functions.values():
                outputs.extend(
                    self.toolchain.compile(
                        crate,
                        crate / "device" / function.name,
                        kernel=function.name,
                    )
                )
            lockfile = crate / "Cargo.lock"
            if lockfile.is_file():
                outputs.append(lockfile)
        pending_manifest = output / ".manifest.json.tmp"
        pending_manifest.write_text(
            json.dumps(
                {
                    "ppy_version": version("ppy-lang"),
                    "ppy_furiosa_version": __version__,
                    "backend_api": self.api_version,
                    "backend_fingerprint": self.fingerprint(),
                    "ppy_artifact_identity": context.identity,
                    "generated_files": [path.relative_to(output).as_posix() for path in outputs],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pending_manifest.replace(manifest)
        outputs.append(manifest)
        return BuildResult(outputs=tuple(outputs))


def create_backend(options: Mapping[str, object] | None = None) -> FuriosaBackend:
    return FuriosaBackend(options)


def _write_crate(crate: Path, component: str, module: IRModule) -> tuple[Path, ...]:
    (crate / "src").mkdir(parents=True, exist_ok=True)
    artifact = emit_rust(module)
    cargo = (
        f'[package]\nname = "ppy_generated_{component}"\nversion = "0.0.0"\n'
        'edition = "2024"\npublish = false\n\n[workspace]\n\n'
        "[package.metadata.furiosa-opt]\n\n"
        '[dependencies]\nfuriosa-opt-std = "=0.6.0"\n\n'
        '[lints.rust]\nunexpected_cfgs = { level = "warn", '
        "check-cfg = ['cfg(backend, values(\"npu\"))', 'cfg(furiosa_opt)'] }\n"
    )
    files = {
        crate / "src/lib.rs": artifact.source,
        crate / "Cargo.toml": cargo,
        crate / "rust-toolchain.toml": '[toolchain]\nchannel = "nightly-2026-05-01"\n',
        crate / "source-map.json": json.dumps(
            {
                "rust_file": "src/lib.rs",
                "locations": [asdict(item) for item in artifact.locations],
            },
            indent=2,
        )
        + "\n",
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    return tuple(files)
