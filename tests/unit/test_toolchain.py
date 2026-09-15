"""Toolchain behavior without a Furiosa SDK or an RNGD device."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from ppy_compiler.backend import BackendError, BackendUnavailable

from ppy_furiosa import toolchain


def test_runner_preserves_arguments_and_environment(tmp_path: Path) -> None:
    result = toolchain.run_command(
        (
            sys.executable,
            "-c",
            "import os,sys; print(os.getcwd()); print(os.environ['VALUE']); "
            "print(sys.argv[1]); print('diagnostic', file=sys.stderr)",
            "a;$(echo unsafe)",
        ),
        cwd=tmp_path,
        env={**os.environ, "VALUE": "explicit"},
        timeout=10,
    )
    assert str(tmp_path) in result.stdout
    assert "explicit\na;$(echo unsafe)" in result.stdout
    assert result.stderr.strip() == "diagnostic"


def test_runner_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="missing executable"):
        toolchain.run_command(
            (str(tmp_path / "missing-command"),), cwd=tmp_path, env=dict(os.environ), timeout=1
        )


def test_runner_error_is_concise(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="exit 7") as caught:
        toolchain.run_command(
            (sys.executable, "-c", "import sys; print('x'*10000, file=sys.stderr); sys.exit(7)"),
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=10,
        )
    assert len(str(caught.value)) < 3000


def test_runner_timeout(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="timed out"):
        toolchain.run_command(
            (sys.executable, "-c", "import time; time.sleep(10)"),
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=0.05,
        )


def test_status_and_fingerprint_cache_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def run(
        argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> toolchain.CommandResult:
        assert cwd.is_absolute() and env and timeout > 0
        calls.append(argv)
        return toolchain.CommandResult(argv, 0, "version 0.6.0\n", "")

    monkeypatch.setattr(toolchain, "run_command", run)
    sdk = toolchain.FuriosaToolchain(cargo="cargo custom", rustc="rustc custom")
    assert sdk.status().available
    assert sdk.fingerprint() == sdk.fingerprint()
    assert calls == [
        ("cargo custom", "--version"),
        ("rustc custom", "--version"),
        ("cargo custom", "furiosa-opt", "--version"),
    ]


def test_missing_sdk_is_status_not_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(
        argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> toolchain.CommandResult:
        assert cwd.is_absolute() and env and timeout > 0
        if "furiosa-opt" in argv:
            raise BackendError("no such command: furiosa-opt")
        return toolchain.CommandResult(argv, 0, "version", "")

    monkeypatch.setattr(toolchain, "run_command", run)
    sdk = toolchain.FuriosaToolchain()
    assert not sdk.status().available
    assert "furiosa-opt" in sdk.status().detail
    assert sdk.fingerprint()


@pytest.mark.parametrize("kernel", ["test::kernel", "broadcast"])
def test_compile_collects_real_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel: str
) -> None:
    crate = tmp_path / "crate with spaces"
    crate.mkdir()
    (crate / "Cargo.toml").write_text('[package]\nname="test"\n', encoding="utf-8")
    output = tmp_path / "output"

    def run(
        argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> toolchain.CommandResult:
        if "compile" in argv:
            assert cwd == crate
            assert argv[:4] == ("cargo", "furiosa-opt", "compile", kernel)
            assert "--exact" not in argv
            assert timeout >= 60
            artifacts = Path(env["FURIOSA_OPT_OUT_DIR"])
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "kernel.bin").write_bytes(b"real compiler output")
            schedule = Path(argv[argv.index("--dump-schedule") + 1])
            schedule.write_text('{"instructions": []}', encoding="utf-8")
        return toolchain.CommandResult(argv, 0, "version", "")

    monkeypatch.setattr(toolchain, "run_command", run)
    files = toolchain.FuriosaToolchain().compile(crate, output, kernel=kernel)
    assert {path.name for path in files} == {"kernel.bin", "schedule.json"}
    assert all(path.is_file() and path.is_absolute() for path in files)


def test_compile_missing_toolchain(tmp_path: Path) -> None:
    sdk = toolchain.FuriosaToolchain(cargo=str(tmp_path / "missing-cargo"))
    with pytest.raises(BackendUnavailable):
        sdk.compile(tmp_path, tmp_path / "output", kernel="test::kernel")


def test_compile_rejects_invalid_kernel(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="kernel"):
        toolchain.FuriosaToolchain().compile(tmp_path, tmp_path, kernel="--dump-ir")


def test_compile_without_artifacts_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nname="test"', encoding="utf-8")

    def run(
        argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> toolchain.CommandResult:
        assert cwd.is_absolute() and env and timeout > 0
        return toolchain.CommandResult(argv, 0, "version", "")

    monkeypatch.setattr(toolchain, "run_command", run)
    with pytest.raises(BackendError, match="artifact"):
        toolchain.FuriosaToolchain().compile(tmp_path, tmp_path / "output", kernel="test::none")


@pytest.mark.parametrize("mode", ["ambiguous", "missing schedule", "stale"])
def test_compile_rejects_unreliable_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nname="test"', encoding="utf-8")
    output = tmp_path / "output"
    (output / "artifacts").mkdir(parents=True)
    (output / "artifacts" / "old.bin").write_bytes(b"old")

    def run(
        argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], timeout: float
    ) -> toolchain.CommandResult:
        assert cwd.is_absolute() and env and timeout > 0
        if "compile" in argv and mode != "stale":
            artifacts = Path(env["FURIOSA_OPT_OUT_DIR"])
            (artifacts / "one.bin").write_bytes(b"new")
            if mode == "ambiguous":
                (artifacts / "two.bin").write_bytes(b"new")
        return toolchain.CommandResult(argv, 0, "version", "")

    monkeypatch.setattr(toolchain, "run_command", run)
    expected = {
        "ambiguous": "multiple artifacts",
        "missing schedule": "no schedule",
        "stale": "no device artifact",
    }[mode]
    with pytest.raises(BackendError, match=expected):
        toolchain.FuriosaToolchain().compile(tmp_path, output, kernel="test::kernel")
    assert (output / "artifacts" / "old.bin").read_bytes() == b"old"
    assert not (output / "artifacts" / "one.bin").exists()
