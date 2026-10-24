#!/usr/bin/env python3
"""Record a compact identity for each upstream source tree.

Two of the three trees ship without Git metadata, so a commit hash cannot
identify them. This writes one file that can: for every tree it records the Git
HEAD when there is one, the file count, the total size, and a single SHA-256
digest over the sorted (relative path, content hash) pairs.

Re-running on an unmodified tree must produce the same digest; that is how an
accidentally edited checkout gets caught.

Usage:
    python3 scripts/checks/source_baseline.py [--out PATH]

Trees are read from the environment (see .env.example):
    GWP_SOURCE_DIR, ROBOTWIN_SOURCE_DIR, WAM_CPP_SOURCE_DIR
Optional: ROBOTWIN_ASSETS_DIR (huge; counted but not hashed by default unless
`--hash-assets` is passed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

TREES = (
    ("giga-world-policy", "GWP_SOURCE_DIR"),
    ("robotwin", "ROBOTWIN_SOURCE_DIR"),
    ("wam.cpp", "WAM_CPP_SOURCE_DIR"),
)

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache"}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root: pathlib.Path):
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def tree_digest(root: pathlib.Path, hash_contents: bool = True):
    entries = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = pathlib.Path(dirpath) / name
            rel = str(p.relative_to(root))
            try:
                size = p.stat().st_size
            except OSError:
                continue
            total += size
            entries.append((rel, sha256_file(p) if hash_contents else str(size)))
    h = hashlib.sha256()
    for rel, sig in entries:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(sig.encode())
        h.update(b"\n")
    return h.hexdigest(), len(entries), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="defaults to $LOCKS_DIR/source_baseline.json")
    ap.add_argument("--hash-assets", action="store_true", help="also digest the assets tree")
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.environ.get("LOCKS_DIR") or os.path.join(os.getcwd(), "locks"),
        "source_baseline.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    report = {"trees": {}, "assets": None, "python": sys.version.split()[0]}
    for label, env in TREES:
        root = os.environ.get(env)
        if not root or not os.path.isdir(root):
            report["trees"][label] = {"env": env, "status": "missing"}
            continue
        p = pathlib.Path(root)
        digest, count, size = tree_digest(p)
        report["trees"][label] = {
            "env": env,
            "git_head": git_head(p),
            "files": count,
            "bytes": size,
            "sha256_tree": digest,
        }
        print(f"{label:20s} files={count:6d} bytes={size:12d} head={report['trees'][label]['git_head']}")

    assets = os.environ.get("ROBOTWIN_ASSETS_DIR")
    if args.hash_assets and assets and os.path.isdir(assets):
        digest, count, size = tree_digest(pathlib.Path(assets), hash_contents=False)
        report["assets"] = {"env": "ROBOTWIN_ASSETS_DIR", "files": count, "bytes": size,
                            "sha256_of_size_list": digest}

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
