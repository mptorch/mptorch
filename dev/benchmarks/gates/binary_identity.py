#!/usr/bin/env python3
"""Is a rebuilt extension the same machine code as the build it replaces?

Output captures and exhaustive sweeps say a change moved no *values*. They
cannot say it moved no *kernels*, and a refactor that is meant to be free --
templating the casts on a carrier, respelling a pun, renaming a type -- is
only free if the instructions are the ones that were there before. This
script compares two builds at the level where that question has an answer:

``so BASE.so NEW.so``
    Every ELF section byte for byte (parsed here, so the non-loaded ones are
    covered too), every member of ``.nv_fatbin`` -- native cubins and the PTX
    kept for other architectures -- every device function's SASS, the
    per-kernel register report, and every host function's disassembly. The
    exit status is 0 only if every loaded section other than the build ID is
    identical, which is the claim; the per-function reports are what find the
    kernel when it is not.
``cubin BASE.cubin NEW.cubin [--ptxas BASE.log NEW.log]``
    A benchmark built with ``nvcc -cubin`` (``timing/gemm_sr_ab.cu``,
    ``timing/gemm_f64_unroll.cu``): its SASS per function, and with
    ``--ptxas`` the registers, stack frame and spills that ``-Xptxas -v``
    printed while building it.
``obj BASE.o NEW.o``
    One host translation unit, compiled twice. The fast way to bisect a host
    difference: rebuild a single ``.o`` against edited headers instead of the
    whole extension.

``.nv_fatbin``'s members are paired by content, not by name: a member's file
name is its index in the fatbin, so a translation unit added to the build
shifts every member after it. A base member with a byte-identical member in
the new build is matched whatever its index, and what is left on either side
is what changed. Device functions are matched by name, and compared as the
multiset of their copies: a kernel that several translation units
instantiate (cub's ``EmptyKernel``) has a copy in each of their cubins, and a
module that adds or drops one changes the copy count, not the code, which is
reported apart from the functions that differ.

A rename that changes a mangled name --
``BinaryKParams`` becoming ``BinaryKParamsT<float>`` -- is undone on the
demangled name with ``--rename 'BinaryKParamsT<float>=BinaryKParams'``, which
may be repeated. Host functions are matched by address order instead, because
``objdump -C`` leaves the longest lambda names mangled and a template argument
shifts every substitution index after it, so no textual rename lines them up;
the linker lays functions out in the same order when the code is the same.
Bodies are compared with addresses, jump and call targets, RIP-relative
displacements and alignment padding removed: a function that moved is not a
function that changed, and a displacement that did change shows up in the
section comparison anyway.

``--show NAME`` prints the normalized diff of every differing function whose
name contains NAME.

Three things this found while the cast layer was templated on its carrier
(``FloatTraits<T>``, 90beb61 and 8112260), all documented in
``mptorch/csrc/common/bit_helper.h``'s carrier comment: a pun behind an
inlined function, a builtin behind a wrapper, and an ``int`` shift respelled
unsigned each move codegen without moving a value, and so does an ``if``
nested inside ``if constexpr``, which only the PTX comparison showed.

Usage, from the repo root::

    cp mptorch/_C.abi3.so /tmp/base.so     # before the change
    # ... change, rebuild ...
    python dev/benchmarks/gates/binary_identity.py so /tmp/base.so mptorch/_C.abi3.so \\
        --rename 'BinaryKParamsT<float>=BinaryKParams'

Needs binutils (``objdump``) and, for the device half, ``cuobjdump`` and
``c++filt`` on ``PATH``; the device half is skipped with a note when they are
missing or the file has no ``.nv_fatbin``.
"""

import argparse
import collections
import difflib
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# shared


def _run(cmd: list[str], stdin: str | None = None, cwd: str | None = None) -> str:
    return subprocess.run(
        cmd, input=stdin, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def _have(*tools: str) -> bool:
    return all(shutil.which(t) for t in tools)


def _renamer(renames: list[str]):
    pairs = []
    for r in renames:
        old, sep, new = r.partition("=")
        if not sep:
            raise SystemExit(f"--rename wants OLD=NEW, got {r!r}")
        pairs.append((old, new))

    def apply(name: str) -> str:
        for old, new in pairs:
            name = name.replace(old, new)
        return name

    return apply


def _digest(lines: list[str]) -> str:
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def _show_diff(name: str, a: list[str], b: list[str]) -> None:
    print(f"\n--- {name}")
    for line in difflib.unified_diff(a, b, lineterm="", n=2):
        if not line.startswith(("---", "+++")):
            print("    " + line)


# ---------------------------------------------------------------------------
# ELF sections, parsed directly: objcopy -O binary drops non-loaded sections


SHF_ALLOC = 0x2
SHT_NOBITS = 8


def _sections(path: str) -> dict[str, tuple[bool, bytes]]:
    """Section name -> (loaded, contents), for a 64-bit little-endian ELF."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise SystemExit(f"{path}: not a 64-bit little-endian ELF")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)

    def header(i: int) -> tuple[int, int, int, int, int]:
        name, kind, flags, _addr, offset, size = struct.unpack_from(
            "<IIQQQQ", data, shoff + i * shentsize
        )
        return name, kind, flags, offset, size

    _, _, _, stroff, _ = header(shstrndx)
    out = {}
    for i in range(1, shnum):
        name_off, kind, flags, offset, size = header(i)
        end = data.index(b"\0", stroff + name_off)
        name = data[stroff + name_off : end].decode()
        body = b"" if kind == SHT_NOBITS else data[offset : offset + size]
        out[name] = (bool(flags & SHF_ALLOC), body)
    return out


def compare_sections(base: str, new: str) -> bool:
    a, b = _sections(base), _sections(new)
    ok = True
    print("sections")
    for name in list(a) + [n for n in b if n not in a]:
        if name not in a or name not in b:
            print(f"  {name:24} only in {'base' if name in a else 'new'}")
            ok = False
            continue
        loaded, body_a = a[name]
        body_b = b[name][1]
        if body_a == body_b:
            continue
        # The build ID hashes the whole file, and .strtab holds local symbol
        # names; neither is code, and a rename changes both.
        benign = name == ".note.gnu.build-id" or not loaded
        if not benign:
            ok = False
        tag = "differs" if not benign else "differs (not code)"
        print(f"  {name:24} {tag}: {len(body_a)} -> {len(body_b)} B")
    print(f"  every loaded section {'identical' if ok else 'NOT identical'}")
    return ok


def compare_fatbin(base: str, new: str) -> bool:
    if ".nv_fatbin" not in _sections(base):
        return True
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        for label, path in (("base", base), ("new", new)):
            d = os.path.join(tmp, label)
            os.makedirs(d)
            # members are named after the input file, so give both the same name
            os.symlink(os.path.abspath(path), os.path.join(d, "binary.so"))
            for flag in ("-xelf", "-xptx"):
                _run(["cuobjdump", flag, "all", "binary.so"], cwd=d)
            dirs.append(d)
        names_a = sorted(n for n in os.listdir(dirs[0]) if n != "binary.so")
        names_b = sorted(n for n in os.listdir(dirs[1]) if n != "binary.so")

        def digest(d: str, n: str) -> str:
            with open(os.path.join(d, n), "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        # every new member, by content; a base member takes an identical one,
        # its own index first, so that an unchanged build pairs index to index
        free: dict[str, list[str]] = collections.defaultdict(list)
        for n in names_b:
            free[digest(dirs[1], n)].append(n)
        unmatched, moved = [], 0
        for n in names_a:
            twins = free.get(digest(dirs[0], n), [])
            if not twins:
                unmatched.append(n)
                continue
            pick = n if n in twins else twins[0]
            twins.remove(pick)
            moved += pick != n
        added = sorted(n for ns in free.values() for n in ns)
    matched = len(names_a) - len(unmatched)
    print(
        f"fatbin members: {len(names_a)} in base, {len(names_b)} in new; {matched} base members "
        f"have a byte-identical one in new" + (f" ({moved} at another index)" if moved else "")
    )
    for n in unmatched:
        print(f"  no identical member in new: {n}")
        ok = False
    for n in added:
        print(f"  only in new: {n}")
        ok = False
    if ok:
        print("  every member identical")
    return ok


# ---------------------------------------------------------------------------
# device


def _sass_functions(path: str, rename) -> dict[str, list[list[str]]]:
    """Every device function's SASS by demangled name, one body per copy: a
    kernel that several translation units instantiate has one in each of
    their cubins."""
    text = _run(["c++filt"], stdin=_run(["cuobjdump", "-sass", path]))
    fns: dict[str, list[list[str]]] = {}
    body: list[str] | None = None
    for line in text.splitlines():
        m = re.match(r"\s*Function : (.*)", line)
        if m:
            body = []
            fns.setdefault(rename(m.group(1).strip()), []).append(body)
            continue
        # A member's header ("Fatbin elf code:", its arch, and "compressed"
        # when the build used --compress-mode=size) follows the last function
        # of the member before it, and is not that function's code.
        if line.startswith("Fatbin "):
            body = None
        if body is None:
            continue
        s = re.sub(r"/\*[0-9a-f]{4,}\*/", "", line)  # the instruction's address
        s = re.sub(r"/\* 0x[0-9a-f]+ \*/", "", s).strip()  # its encoding
        if s:
            body.append(s)
    return fns


def compare_device_functions(base: str, new: str, rename, show: str | None) -> bool:
    a, b = _sass_functions(base, rename), _sass_functions(new, rename)
    same, recopied, differ = 0, [], []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            differ.append((name, "only in " + ("base" if name in a else "new")))
            continue
        ca = collections.Counter(_digest(body) for body in a[name])
        cb = collections.Counter(_digest(body) for body in b[name])
        if ca == cb:
            same += 1
        elif set(ca) == set(cb):
            # the same code, in more or fewer modules
            same += 1
            recopied.append((name, f"{sum(ca.values())} -> {sum(cb.values())} copies"))
        else:
            # a copy the new build has fewer of, against one it has more of
            gone, came = ca - cb, cb - ca
            old = next((x for x in a[name] if _digest(x) in gone), a[name][0])
            new_ = next(x for x in b[name] if _digest(x) in came)
            differ.append((name, f"{len(old)} -> {len(new_)} instructions"))
            if show and show in name:
                _show_diff(name, old, new_)
    print(f"device functions (SASS): {same} identical, {len(differ)} differ")
    if recopied:
        print(f"  of the identical, {len(recopied)} in a different number of modules:")
        for name, what in recopied:
            print(f"    {what:26} {name[:150]}")
    for name, what in differ:
        print(f"  {what:28} {name[:150]}")
    return not differ


def _resource_usage(path: str, rename) -> dict[str, list[str]]:
    """Each kernel's register report, one per copy (see `_sass_functions`)."""
    text = _run(["c++filt"], stdin=_run(["cuobjdump", "-res-usage", path]))
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"\s*Function (.*):$", line)
        if m:
            name = rename(m.group(1))
        elif name and "REG:" in line:
            out.setdefault(name, []).append(" ".join(line.split()))
            name = None
    return out


def compare_resource_usage(base: str, new: str, rename) -> bool:
    a, b = _resource_usage(base, rename), _resource_usage(new, rename)
    # a copy-count change alone is the same report, as in compare_device_functions
    differ = [n for n in sorted(set(a) | set(b)) if set(a.get(n, [])) != set(b.get(n, []))]
    print(f"device resource usage: {len(a)} kernels, {len(differ)} differ")
    for n in differ:
        was = " / ".join(sorted(set(a[n]))) if n in a else "(absent)"
        now = " / ".join(sorted(set(b[n]))) if n in b else "(absent)"
        print(f"  {was}  ->  {now}\n    {n[:150]}")
    return not differ


def _ptxas_report(path: str, rename) -> dict[str, list[str]]:
    with open(path) as f:
        text = f.read()
    mangled: list[str] = []
    lines: dict[str, list[str]] = collections.defaultdict(list)
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            mangled.append(m.group(1))
        elif mangled and ("Used " in line or "stack frame" in line):
            lines[mangled[-1]].append(" ".join(line.replace("ptxas info    :", "").split()))
    names = _run(["c++filt"], stdin="\n".join(mangled)).splitlines()
    return {rename(n): lines[m] for m, n in zip(mangled, names, strict=True)}


def compare_ptxas(base_log: str, new_log: str, rename) -> bool:
    a, b = _ptxas_report(base_log, rename), _ptxas_report(new_log, rename)
    differ = [n for n in sorted(set(a) | set(b)) if a.get(n) != b.get(n)]
    print(f"ptxas reports: {len(a)} kernels, {len(differ)} differ")
    for n in differ:
        print(f"  {n[:150]}\n    base: {a.get(n)}\n    new:  {b.get(n)}")
    return not differ


# ---------------------------------------------------------------------------
# host


def _host_functions(path: str) -> list[tuple[str, list[str]]]:
    text = _run(["objdump", "-d", "--no-show-raw-insn", "-C", path])
    fns: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            fns.append((m.group(1), []))
            continue
        if not fns or not line.strip() or not re.match(r"^\s*[0-9a-f]+:", line):
            continue
        s = re.sub(r"^\s*[0-9a-f]+:\s*", "", line)
        if re.match(r"(nop|cs nop|xchg\s+%ax,%ax|data16)", s):
            continue  # alignment padding: depends on where the function landed
        s = re.sub(r"\b[0-9a-f]+\s+<.*>$", "<target>", s)  # call and jump targets
        s = re.sub(r"0x[0-9a-f]+\(%rip\)", "X(%rip)", s)
        fns[-1][1].append(s.strip())
    return [(n, body) for n, body in fns if not n.endswith("@plt")]


def compare_host_functions(base: str, new: str, show: str | None) -> bool:
    a, b = _host_functions(base), _host_functions(new)
    ha, hb = [_digest(body) for _, body in a], [_digest(body) for _, body in b]
    matcher = difflib.SequenceMatcher(None, ha, hb, autojunk=False)
    same, differ, unpaired = 0, [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            same += i2 - i1
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            fa = a[i1 + k] if i1 + k < i2 else None
            fb = b[j1 + k] if j1 + k < j2 else None
            if fa is None or fb is None:
                side, fn = ("base", fa) if fb is None else ("new", fb)
                assert fn is not None
                unpaired.append(f"{side}: {fn[0]}")
            else:
                changed = sum(
                    1
                    for line in difflib.unified_diff(fa[1], fb[1], lineterm="", n=0)
                    if line[:1] in "+-" and not line.startswith(("+++", "---"))
                )
                differ.append((fa[0], f"{len(fa[1])} -> {len(fb[1])} insns, {changed} lines"))
                if show and show in fa[0]:
                    _show_diff(fa[0], fa[1], fb[1])
    print(f"host functions: {same} identical, {len(differ)} differ, {len(unpaired)} unpaired")
    for name, what in differ:
        print(f"  {what:32} {name[:150]}")
    for name in unpaired:
        print(f"  unpaired {name[:150]}")
    # A change that *adds* instantiations -- a dtype joining a dispatch --
    # interleaves new functions with the old ones, and the address-order
    # pairing above then reports the old ones as differing from whatever they
    # were paired with. The question for such a change is whether every body
    # the base build had is still in the new one, somewhere, unchanged.
    missing = collections.Counter(ha) - collections.Counter(hb)
    print(
        f"host bodies: {len(ha) - sum(missing.values())} of the base's {len(ha)} "
        f"found unchanged in new ({len(hb)} there)"
    )
    for (name, _), digest in zip(a, ha, strict=True):
        if missing[digest] > 0:
            missing[digest] -= 1
            print(f"  missing from new: {name[:150]}")
    return not differ and not unpaired


# ---------------------------------------------------------------------------


def _verdict(results: Iterable[bool]) -> int:
    ok = all(results)
    print("\nIDENTICAL" if ok else "\nNOT IDENTICAL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("so", "cubin", "obj"):
        s = sub.add_parser(name)
        s.add_argument("base")
        s.add_argument("new")
        s.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
        s.add_argument("--show", metavar="NAME")
        if name == "cubin":
            s.add_argument("--ptxas", nargs=2, metavar=("BASE_LOG", "NEW_LOG"))
    args = p.parse_args()
    rename = _renamer(args.rename)
    device = _have("cuobjdump", "c++filt")

    if args.cmd == "obj":
        return _verdict([compare_host_functions(args.base, args.new, args.show)])
    if args.cmd == "cubin":
        if not device:
            raise SystemExit("cubin needs cuobjdump and c++filt on PATH")
        results = [compare_device_functions(args.base, args.new, rename, args.show)]
        if args.ptxas:
            base_log, new_log = args.ptxas
            results.append(compare_ptxas(base_log, new_log, rename))
        return _verdict(results)

    # The sections are the claim; the per-function reports locate a failure,
    # and a difference they report inside an identical section is a
    # normalization artifact, so they do not decide the exit status.
    sections_ok = compare_sections(args.base, args.new)
    if device and ".nv_fatbin" in _sections(args.base):
        compare_fatbin(args.base, args.new)
        compare_device_functions(args.base, args.new, rename, args.show)
        compare_resource_usage(args.base, args.new, rename)
    else:
        print("device half skipped: no .nv_fatbin, or cuobjdump/c++filt not on PATH")
    compare_host_functions(args.base, args.new, args.show)
    return _verdict([sections_ok])


if __name__ == "__main__":
    sys.exit(main())
