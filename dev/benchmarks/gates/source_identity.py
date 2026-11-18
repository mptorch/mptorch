#!/usr/bin/env python3
"""Did a change to the C++ sources touch only comments, or move code without changing it?

A comment edit in C++ is not free to check by building: every ``TORCH_CHECK``
records ``__LINE__``, so a host object below an edited comment changes even
though its behaviour does not, and on macOS the Metal sources are embedded in
the extension with their comments. This script answers the question on the
sources instead, before anything is built: it compares each file with comments
removed (by ``g++ -fpreprocessed``, which removes them as the compiler would and
keeps every directive, string and raw string) and whitespace normalized.

``diff BASE [PATHS...]``
    Every C++ source under ``mptorch/csrc`` and ``dev/benchmarks`` (or under
    PATHS) that differs from revision BASE: modified files, files renamed or
    moved (matched by name to a file deleted since BASE), and new and deleted
    files, which have no counterpart and are listed. Each pair must be identical once
    comments are removed. With ``--resolve-includes``, an ``#include "..."``
    line compares by the file it finds rather than by its text, so a file
    moved into a subdirectory whose includes gained a ``../`` still compares
    as identical.
``split BASE OLD NEW [NEW...]``
    One file split into several: every top-level definition of OLD at BASE
    (each function, class or declaration, comments removed) must appear in
    exactly one NEW file, unchanged, and the NEW files must define nothing
    else. ``#include`` and ``using`` lines are listed, not compared, since a
    split changes which each file needs.

The exit status is the number of files (or definitions) that differ.

Usage, from the repo root::

    python dev/benchmarks/gates/source_identity.py diff HEAD
    python dev/benchmarks/gates/source_identity.py diff HEAD --resolve-includes
    python dev/benchmarks/gates/source_identity.py split HEAD mptorch/csrc/meta_gemm.cpp \\
        mptorch/csrc/meta/*_entry.cpp

Needs ``g++`` on ``PATH``. Takes seconds.
"""

import argparse
import difflib
import posixpath
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh", ".mm", ".metal"}
DEFAULT_ROOTS = ["mptorch/csrc", "dev/benchmarks"]

_PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$", re.M)
_INCLUDE = re.compile(r'^#\s*include\s*"([^"]+)"\s*$')
_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return done.stdout


def at_revision(rev: str, path: str) -> str:
    return git("show", f"{rev}:{path}")


def strip_comments(text: str) -> list[str]:
    """Return the lines of ``text`` with comments removed and whitespace
    normalized: blank lines dropped, and runs of spaces outside string and
    character literals collapsed to one."""
    # g++ drops `#pragma once` from the file it is given, so keep it as a
    # pragma it does not know, which it passes through.
    text = _PRAGMA_ONCE.sub("#pragma mptorch_source_identity_once", text)
    out = subprocess.run(
        ["g++", "-x", "c++", "-std=c++20", "-fpreprocessed", "-dD", "-E", "-P", "-w", "-"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = []
    for line in out.splitlines():
        parts, last = [], 0
        for m in _LITERAL.finditer(line):
            parts += [re.sub(r"\s+", " ", line[last : m.start()]), m.group(0)]
            last = m.end()
        parts.append(re.sub(r"\s+", " ", line[last:]))
        norm = "".join(parts).strip()
        if norm:
            lines.append(norm)
    return lines


def resolve_includes(
    lines: list[str], path: str, exists: Callable[[str], bool], renames: dict[str, str]
) -> list[str]:
    """Replace each quoted include by the repo path of the file it finds:
    first beside the including file ``path``, then under ``mptorch/csrc`` (the
    build's include path). ``exists`` says whether a repo path exists in the
    tree ``path`` is from, and ``renames`` maps a path to where it moved, so a
    base file's include of a header that has since moved compares equal to the
    working tree's include of it."""
    out = []
    for line in lines:
        m = _INCLUDE.match(line)
        if m:
            for base in (posixpath.dirname(path), "mptorch/csrc"):
                found = posixpath.normpath(posixpath.join(base, m.group(1)))
                if exists(found):
                    line = f"#include <{renames.get(found, found)}>"
                    break
        out.append(line)
    return out


def changed_files(
    base: str, roots: list[str]
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Return ``(pairs, new, gone)``: (old path, new path) for every modified
    or moved C++ file, and the paths of new files and of deleted files that
    have no counterpart."""
    pairs, deleted, added = [], [], []
    for line in git("diff", "--name-status", "-M", base, "--", *roots).splitlines():
        status, *paths = line.split("\t")
        if not any(Path(p).suffix in SUFFIXES for p in paths):
            continue
        if status.startswith("R"):
            pairs.append((paths[0], paths[1]))
        elif status == "M":
            pairs.append((paths[0], paths[0]))
        elif status == "D":
            deleted.append(paths[0])
        elif status == "A":
            added.append(paths[0])
    added += [
        p
        for p in git("ls-files", "--others", "--exclude-standard", "--", *roots).splitlines()
        if Path(p).suffix in SUFFIXES
    ]
    by_name: dict[str, list[str]] = {}
    for p in deleted:
        by_name.setdefault(Path(p).name, []).append(p)
    new = []
    for p in added:
        old = by_name.get(Path(p).name)
        if old and len(old) == 1:
            pairs.append((old.pop(), p))
        else:
            new.append(p)
    gone = [p for paths in by_name.values() for p in paths]
    return pairs, new, gone


def cmd_diff(args: argparse.Namespace) -> int:
    pairs, new, gone = changed_files(args.base, args.paths or DEFAULT_ROOTS)
    base_files = set(git("ls-tree", "-r", "--name-only", args.base).splitlines())
    # every move since BASE, not only those under PATHS, since an include can
    # name a file outside them
    renames = {old: now for old, now in changed_files(args.base, ["."])[0] if old != now}
    bad = 0
    for old, now in sorted(pairs):
        a = strip_comments(at_revision(args.base, old))
        b = strip_comments((ROOT / now).read_text())
        if args.resolve_includes:
            a = resolve_includes(a, old, base_files.__contains__, renames)
            b = resolve_includes(b, now, lambda p: (ROOT / p).exists(), {})
        moved = f"  (moved from {old})" if old != now else ""
        if a == b:
            print(f"identical  {now}{moved}")
            continue
        bad += 1
        print(f"DIFFERS    {now}{moved}")
        for d in list(difflib.unified_diff(a, b, lineterm="", n=0))[2:14]:
            print(f"    {d}")
    for p in new:
        print(f"new        {p}")
    for p in gone:
        print(f"deleted    {p}")
    print(f"{len(pairs)} changed files compared, {bad} differ, {len(new)} new, {len(gone)} deleted")
    return bad


def definitions(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split comment-free source into its top-level items, each ending in a
    `;` or a `}` at brace depth 0, and its `#include`/`using` lines."""
    items, heads, current, depth = [], [], [], 0
    for line in lines:
        if depth == 0 and not current and line.startswith(("#include", "using ")):
            heads.append(line)
            continue
        current.append(line)
        for m in _LITERAL.sub('""', line):
            depth += (m == "{") - (m == "}")
        if depth == 0 and (line.endswith("}") or line.endswith(";")):
            items.append("\n".join(current))
            current = []
    if current:
        items.append("\n".join(current))
    return items, heads


def cmd_split(args: argparse.Namespace) -> int:
    old_items, old_heads = definitions(strip_comments(at_revision(args.base, args.old)))
    new_items: Counter[str] = Counter()
    for p in args.new:
        items, heads = definitions(strip_comments(Path(p).read_text()))
        new_items.update(items)
        print(f"{p}: {len(items)} definitions; heads: {heads}")
    old = Counter(old_items)
    missing, extra = old - new_items, new_items - old
    for item in missing:
        print(f"MISSING or changed: {item.splitlines()[0][:100]}")
    for item in extra:
        print(f"NEW or changed:     {item.splitlines()[0][:100]}")
    print(f"{args.old}: {len(old_items)} definitions, heads {old_heads}")
    n_old, n_missing, n_extra = sum(old.values()), sum(missing.values()), sum(extra.values())
    print(f"{n_old} old definitions, {n_missing} missing, {n_extra} extra")
    return sum(missing.values()) + sum(extra.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="changed files against BASE, comments removed")
    d.add_argument("base")
    d.add_argument("paths", nargs="*")
    d.add_argument("--resolve-includes", action="store_true")
    s = sub.add_parser("split", help="one file split into several")
    s.add_argument("base")
    s.add_argument("old")
    s.add_argument("new", nargs="+")
    args = ap.parse_args()
    return min(cmd_diff(args) if args.cmd == "diff" else cmd_split(args), 255)


if __name__ == "__main__":
    sys.exit(main())
