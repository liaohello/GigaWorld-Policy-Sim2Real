"""Load `$REPO_ROOT/.env` for Python entrypoints.

The shell scripts source `scripts/lib/env.sh`, which does the same thing. Python
entrypoints that can be run directly (`python3 scripts/evaluate/smoke_env.py`)
need their own loader, otherwise they work only when the caller remembered to
export the variables first — and the failure mode is a confusing `None` deep
inside a path join.

Deliberately dependency-free: this must work in the simulation environment,
which has no python-dotenv.
"""

from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_env(path: str | pathlib.Path | None = None, *, override: bool = False) -> dict:
    """Read a .env file into os.environ. Existing values win unless override.

    Supports the subset actually used by this project: `KEY=VALUE`, optional
    `export ` prefix, `#` comments, and single/double quoted values. No
    interpolation, no multiline values — if `.env` ever needs those, this
    loader should fail rather than guess.
    """
    env_path = pathlib.Path(path) if path else REPO_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def require(*names: str) -> dict[str, str]:
    """Return the named variables, raising a readable error if any is missing."""
    load_env()
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            "missing environment variables: "
            + ", ".join(missing)
            + "\n  copy .env.example to .env and fill them in, or export them directly"
        )
    return {n: os.environ[n] for n in names}
