"""Command sandbox holding the live context file.

``LocalSandbox`` runs bash in a private working directory on the host (Git
Bash on Windows). It is *not* a security boundary: use it only with models
and tasks you trust, or swap in a container-backed sandbox with the same
interface (``run``, ``write_context``, ``read_context``, ``ctx_path``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CTX_SUBDIR = ".live_ctx"
CTX_NAME = "LIVE_CTX_MAIN.txt"


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def output(self) -> str:
        out = self.stdout
        if self.stderr.strip():
            out = (out + ("\n" if out and not out.endswith("\n") else "") + self.stderr)
        return out


class LocalSandbox:
    def __init__(self, workdir: str | os.PathLike | None = None, shell: str | None = None):
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="clm_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.shell = shell or shutil.which("bash") or "/bin/bash"
        (self.workdir / CTX_SUBDIR).mkdir(exist_ok=True)
        self._bin = self.workdir / ".clm_bin"
        self._bin.mkdir(exist_ok=True)
        self._install_python_shims()

    # -------------------------------------------------------------- context
    @property
    def ctx_file(self) -> Path:
        return self.workdir / CTX_SUBDIR / CTX_NAME

    @property
    def ctx_path(self) -> str:
        """Path as written in prompts; forward slashes work in bash and Python on every OS."""
        return self.ctx_file.resolve().as_posix()

    def write_context(self, text: str) -> None:
        self.ctx_file.write_text(text, encoding="utf-8", newline="\n")

    def read_context(self) -> str | None:
        try:
            return self.ctx_file.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None

    # -------------------------------------------------------------- commands
    def env(self) -> dict:
        env = dict(os.environ)
        env["PATH"] = str(self._bin) + os.pathsep + env.get("PATH", "")
        env["CLM_CTX"] = self.ctx_path
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run(self, command: str, timeout: float = 180.0) -> CommandResult:
        try:
            p = subprocess.run([self.shell, "-c", command], cwd=self.workdir, env=self.env(),
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return CommandResult(_dec(e.stdout), _dec(e.stderr) + f"\n[command timed out after {timeout}s]",
                                 124, timed_out=True)
        return CommandResult(_dec(p.stdout), _dec(p.stderr), p.returncode)

    def path(self, rel: str) -> Path:
        return self.workdir / rel

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _install_python_shims(self) -> None:
        # Models write `python3 - <<'EOF'`; on Windows `python3` is often a Store stub.
        if os.name != "nt" and shutil.which("python3"):
            return
        exe = Path(sys.executable).as_posix()
        for name in ("python3", "python"):
            shim = self._bin / name
            shim.write_text(f'#!/bin/sh\nexec "{exe}" "$@"\n', encoding="utf-8", newline="\n")
            shim.chmod(0o755)


def _dec(b) -> str:
    if b is None:
        return ""
    if isinstance(b, str):
        return b
    return b.decode("utf-8", errors="replace")
