from __future__ import annotations

import os
import subprocess
from typing import Iterable, Mapping


def run_checked(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run a subprocess; on failure, raise RuntimeError with captured stdout/stderr."""
    try:
        merged_env = None
        if env is not None:
            merged_env = os.environ.copy()
            merged_env.update(dict(env))
        subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = [
            "Command failed:",
            "  " + " ".join(cmd),
        ]
        if e.stdout:
            msg.append("--- stdout ---")
            msg.append(e.stdout.strip())
        if e.stderr:
            msg.append("--- stderr ---")
            msg.append(e.stderr.strip())
        raise RuntimeError("\n".join(msg)) from e


def ensure_list(cmd: str | Iterable[str]) -> list[str]:
    if isinstance(cmd, str):
        return [cmd]
    return list(cmd)
