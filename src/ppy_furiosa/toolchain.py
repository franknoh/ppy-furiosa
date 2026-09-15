"""Typed, shell-free access to the Furiosa Rust compiler toolchain.

Command syntax targets cargo-furiosa-opt 0.6.0. The SDK launcher supplies the
driver's Rust shared-library environment; invoking the driver directly does not.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ppy_compiler.backend import BackendError, BackendUnavailable, ToolchainStatus

_PROBE_TIMEOUT = 15.0
_COMPILE_TIMEOUT = 1800.0
_DIAGNOSTIC_LIMIT = 2000
_KERNEL_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*")


@dataclass(frozen=True)
class CommandResult:
    """Captured output from a successful command invocation."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_command(
    argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], timeout: float
) -> CommandResult:
    """Execute an argv vector with explicit process state and checked status."""
    if not argv or not argv[0] or timeout <= 0:
        raise BackendError("invalid toolchain command or timeout")
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise BackendUnavailable(f"missing executable {argv[0]!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BackendError(f"{argv[0]!r} timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise BackendError(f"cannot execute {argv[0]!r}: {exc}") from exc
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()[-_DIAGNOSTIC_LIMIT:]
        raise BackendError(f"{argv[0]!r} failed (exit {result.returncode}): {diagnostic}")
    return CommandResult(argv, result.returncode, result.stdout, result.stderr)


@dataclass(frozen=True)
class FuriosaToolchain:
    """Configured compiler commands with lazily cached version probes.

    Status checks command availability. SDK dependency resolution and driver
    loading are checked by compilation, without requiring an RNGD device.
    Create a new instance after installing or changing a toolchain.
    """

    cargo: str = "cargo"
    rustc: str = "rustc"

    @cached_property
    def _probes(self) -> tuple[ToolchainStatus, tuple[str, ...]]:
        versions: list[str] = []
        failures: list[str] = []
        env = self._environment()
        for label, argv in (
            ("cargo", (self.cargo, "--version")),
            ("rustc", (self.rustc, "--version")),
            ("furiosa-opt", (self.cargo, "furiosa-opt", "--version")),
        ):
            try:
                result = run_command(argv, cwd=Path.cwd(), env=env, timeout=_PROBE_TIMEOUT)
                version = (result.stdout or result.stderr).strip()
                if not version:
                    raise BackendError(f"{label} returned no version")
                versions.append(f"{label}: {version}")
            except BackendError as exc:
                failures.append(f"{label}: {exc}")
        detail = "; ".join(failures or versions)
        return ToolchainStatus(not failures, detail), tuple(versions + failures)

    def _environment(self) -> dict[str, str]:
        return {**os.environ, "RUSTC": self.rustc, "CARGO_TERM_COLOR": "never"}

    def status(self) -> ToolchainStatus:
        """Report absent tools as an unavailable status, never an import error."""
        return self._probes[0]

    def fingerprint(self) -> str:
        """Identify commands, versions, and environment that affect compilation."""
        state = {
            "schema": 1,
            "cargo": self.cargo,
            "rustc": self.rustc,
            "probes": self._probes[1],
            "environment": {
                key: os.environ.get(key, "")
                for key in (
                    "RUSTUP_TOOLCHAIN",
                    "RUSTFLAGS",
                    "CARGO_ENCODED_RUSTFLAGS",
                    "RUSTC_WRAPPER",
                    "RUSTC_WORKSPACE_WRAPPER",
                    "CARGO_HOME",
                    "RUSTUP_HOME",
                )
            },
        }
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def compile(self, crate: Path, output: Path, *, kernel: str) -> tuple[Path, ...]:
        """Compile one device function and publish its real outputs.

        Names are relative to the crate, such as ``broadcast`` or ``ops::linear``.
        Furiosa 0.6.0 filters are substring matches. An ambiguous filter producing
        several binaries is rejected. Staging prevents old files from satisfying
        the successful-artifact checks after a failed or empty compilation.
        """
        if _KERNEL_NAME.fullmatch(kernel) is None:
            raise BackendError(f"invalid Furiosa kernel name: {kernel!r}")
        status = self.status()
        if not status.available:
            raise BackendUnavailable(status.detail)
        crate = crate.resolve()
        output = output.resolve()
        if not (crate / "Cargo.toml").is_file():
            raise BackendError(f"Furiosa crate has no Cargo.toml: {crate}")
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".furiosa-", dir=output) as directory:
            staging = Path(directory)
            artifacts = staging / "artifacts"
            schedules = staging / "schedules"
            artifacts.mkdir()
            schedules.mkdir()
            env = {**self._environment(), "FURIOSA_OPT_OUT_DIR": str(artifacts)}
            run_command(
                (
                    self.cargo,
                    "furiosa-opt",
                    "compile",
                    kernel,
                    "--dump-schedule",
                    str(schedules / "schedule.json"),
                ),
                cwd=crate,
                env=env,
                timeout=_COMPILE_TIMEOUT,
            )
            binaries = tuple(artifacts.rglob("*.bin"))
            if not binaries:
                raise BackendError(f"furiosa-opt produced no device artifact for kernel {kernel!r}")
            if len(binaries) != 1:
                raise BackendError(f"Furiosa kernel filter {kernel!r} matched multiple artifacts")
            if not any(path.is_file() for path in schedules.rglob("*")):
                raise BackendError(f"furiosa-opt produced no schedule for kernel {kernel!r}")
            files: list[Path] = []
            for source in sorted(path for path in staging.rglob("*") if path.is_file()):
                destination = output / source.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                files.append(destination)
            return tuple(files)
