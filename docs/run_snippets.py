"""Run every documentation snippet and record what it printed.

The pages under ``docs/source`` include each ``docs/snippets/*.py`` file and,
next to it, the ``*.out`` file this script writes -- so every code block in
the documentation is backed by a real run, and re-running this script after a
change to the library is how the recorded outputs are kept honest.

    python3 docs/run_snippets.py            # all snippets
    python3 docs/run_snippets.py formats    # only those whose name contains it

Each snippet runs in its own interpreter with ``docs/snippets`` as the working
directory, so a relative path inside one (the MNIST download in the tutorial)
lands there. Stdout and stderr are captured together; a non-zero exit fails
the run, and the failing output is left in place for inspection.
"""

import subprocess
import sys
import time
from pathlib import Path

SNIPPETS = Path(__file__).resolve().parent / "snippets"


def main() -> int:
    patterns = sys.argv[1:]
    scripts = sorted(SNIPPETS.rglob("*.py"))
    if patterns:
        scripts = [s for s in scripts if any(p in str(s.relative_to(SNIPPETS)) for p in patterns)]
    failed = []
    for script in scripts:
        rel = script.relative_to(SNIPPETS)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=SNIPPETS,
            capture_output=True,
            text=True,
        )
        out = proc.stdout + proc.stderr
        script.with_suffix(".out").write_text(out)
        dt = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"{str(rel):<48}{dt:>8.1f}s  {status}")
        if proc.returncode != 0:
            failed.append(rel)
            print(out)
    if failed:
        print(f"\n{len(failed)} snippet(s) failed: {', '.join(map(str, failed))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
