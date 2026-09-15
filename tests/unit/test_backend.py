"""Backend boundaries refuse unsupported configuration and IR explicitly."""

import json
from pathlib import Path

import pytest
from ppy_compiler.backend import BackendConfig, BackendContext, BackendError
from ppy_compiler.ir import IRModule
from ppy_compiler.ir.codec import decode, encode

from ppy_furiosa.backend import create_backend
from ppy_furiosa.physical import broadcast_module, make_registry
from ppy_furiosa.toolchain import FuriosaToolchain


def context() -> BackendContext:
    return BackendContext(Path.cwd(), None, BackendConfig("furiosa"), 2, "rngd", make_registry())


@pytest.mark.parametrize("options", [{"targte": "rngd"}, {"target": "cuda"}, {"cargo": 12}])
def test_invalid_options_are_not_ignored(options: dict[str, object]) -> None:
    with pytest.raises(BackendError):
        create_backend(options)


def test_emits_without_sdk_and_refuses_other_formats() -> None:
    backend = create_backend({"cargo": "does-not-exist"})
    assert backend.api_version == 1
    assert backend.emit_formats()[0].name == "furiosa-rust"
    assert "CustomBroadcast" in backend.emit(broadcast_module(), "furiosa-rust", context())
    assert not backend.toolchain_status().available
    with pytest.raises(BackendError, match="format"):
        backend.emit(broadcast_module(), "llvm", context())


def test_empty_ir_is_not_a_successful_build() -> None:
    with pytest.raises(BackendError, match="empty"):
        create_backend({}).validate(IRModule("empty"), context())


def test_backend_activation_supports_ppy031_default_registry_reader() -> None:
    create_backend({})
    encoded = encode(broadcast_module(), make_registry())
    assert decode(encoded).name == "broadcast"


def test_relative_output_manifest_handles_absolute_toolchain_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def compile_stub(
        _self: FuriosaToolchain, crate: Path, output: Path, *, kernel: str
    ) -> tuple[Path, ...]:
        assert (crate / "src/lib.rs").is_file() and kernel == "broadcast"
        output.mkdir(parents=True, exist_ok=True)
        artifact = output.resolve() / "broadcast.bin"
        artifact.write_bytes(b"test artifact")
        return (artifact,)

    monkeypatch.setattr(FuriosaToolchain, "compile", compile_stub)
    monkeypatch.chdir(tmp_path)
    backend = create_backend({"cargo": "missing-command"})
    result = backend.build({"broadcast": broadcast_module()}, Path("output"), context())
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert all((tmp_path / "output" / path).is_file() for path in manifest["generated_files"])
    assert all(path.is_absolute() and path.is_file() for path in result.outputs)


def test_failed_rebuild_invalidates_previous_success_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_compile(
        _self: FuriosaToolchain,
        crate: Path,
        output: Path,
        *,
        kernel: str,
    ) -> tuple[Path, ...]:
        assert crate.is_dir() and output.name == kernel == "broadcast"
        raise BackendError("test compiler failure")

    (tmp_path / "manifest.json").write_text('{"generated_files": ["old.bin"]}')
    (tmp_path / "old.bin").write_bytes(b"old")
    monkeypatch.setattr(FuriosaToolchain, "compile", failed_compile)
    with pytest.raises(BackendError, match="test compiler failure"):
        create_backend({}).build({"broadcast": broadcast_module(32)}, tmp_path, context())
    assert not (tmp_path / "manifest.json").exists()
