#!/usr/bin/env python3
"""Fail if the *indexed* content contains private paths, addresses, or SSH data.

This repository is public. `AGENTS.md` forbids committing home directories,
routable addresses, SSH invocations, credentials, and machine identifiers.

Two details matter and both were wrong in an earlier shell version:

1. **Scan the index, not the working tree.** `git ls-files` lists tracked paths,
   but reading those paths from disk scans whatever is on disk *now*. A file
   staged with a secret and then edited locally would pass. This script reads
   the blob at `:<path>` (the staged/HEAD content) by default.

2. **Exceptions apply to a match, not to a line.** Skipping a whole line that
   happens to contain `127.0.0.1` also hides any other violation on that line.
   Each pattern decides per match here.

Usage:
    python3 scripts/checks/no_private_paths.py              # staged content
    python3 scripts/checks/no_private_paths.py --rev HEAD   # a commit
    python3 scripts/checks/no_private_paths.py --working-tree
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ALLOW_FILES = {
    # Attribution names a person; that is not a machine path.
    "LICENSE",
    # The checker necessarily contains the patterns it searches for.
    "scripts/checks/no_private_paths.py",
}

# A pinned requirement legitimately contains four dotted numbers
# (`opencv-python==4.11.0.86`), which the IPv4 pattern would otherwise flag.
REQUIREMENT_LINE = re.compile(r"^[A-Za-z0-9_.+\-]+==")

LOOPBACK_OK = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}

OCTET = r"(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("home_path", re.compile(r"/(?:home|Users)/[A-Za-z0-9_.\-]+")),
    ("asset_root", re.compile(r"dataAssets")),
    ("ipv4", re.compile(rf"\b{OCTET}(?:\.{OCTET}){{3}}\b")),
    ("ssh_key", re.compile(r"ssh\s+-i\b")),
    ("ssh_dir", re.compile(r"\.ssh/")),
    ("ssh_port", re.compile(r"ssh\b[^\n]*-p\s*\d{4,5}")),
    ("private_key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def iter_indexed():
    for raw in _git("ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        path = raw.decode()
        yield path, _git("show", f":{path}")


def iter_working_tree():
    for path in _git("ls-files", "-z").split(b"\0"):
        if not path:
            continue
        name = path.decode()
        try:
            yield name, pathlib.Path(name).read_bytes()
        except FileNotFoundError:
            continue


def iter_rev(rev: str):
    paths = _git("ls-tree", "-r", "--name-only", "-z", rev).split(b"\0")
    for raw in paths:
        if not raw:
            continue
        name = raw.decode()
        yield name, _git("show", f"{rev}:{name}")


def _allowed(name: str, kind: str, match: re.Match, line: str, offset: int) -> bool:
    if kind == "ipv4":
        if match.group(0) in LOOPBACK_OK:
            return True
        if REQUIREMENT_LINE.match(line.strip()):
            return True
    return False


def scan(entries) -> int:
    violations = 0
    for name, blob in entries:
        if name in ALLOW_FILES:
            continue
        if b"\0" in blob[:8192]:
            continue  # binary; no text patterns to find
        text = blob.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in PATTERNS:
                for match in pattern.finditer(line):
                    if _allowed(name, kind, match, line, match.start()):
                        continue
                    col = match.start() + 1
                    print(f"VIOLATION {name}:{lineno}:{col}  [{kind}] {match.group(0)!r}")
                    violations += 1
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev", default=None, help="scan a commit instead of the index")
    ap.add_argument("--working-tree", action="store_true", help="scan files on disk")
    args = ap.parse_args()

    if args.working_tree:
        source, entries = "working tree", iter_working_tree()
    elif args.rev:
        source, entries = f"rev {args.rev}", iter_rev(args.rev)
    else:
        source, entries = "index", iter_indexed()

    violations = scan(entries)
    if violations:
        print()
        print(f"{violations} violation(s) in the {source}.")
        print("Replace the value with an environment variable documented in .env.example,")
        print("or drop the line. Do not 'fix' this by adding an allowlist entry.")
        return 1

    print(f"ok: no private paths, addresses, or SSH material in the {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
