#!/usr/bin/env python
"""Check every `<!-- check: chNN.key = value ± tol -->` tag in the guide against
the value worked_examples.py actually computes.

    ~/.venvs/lensjudge/bin/python site/guide_src/verify_numbers.py

This is the guide's credibility gate, and it is cheap: one process, all examples
evaluated once, every tag in every chapter compared. (Shelling out per tag takes
minutes; this takes a second.)

Why it exists: this repo's own final report states "~17 sigma" in its abstract,
its README and a commit message. The quoted uncertainties give 9.4, 9.7 or 41 —
never 17 — and the report's own footnote says ~9.5. Nobody did the division. A
guide that repeats numbers it has not divided is worth less than no guide.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import worked_examples as W

HERE = Path(__file__).resolve().parent
GUIDE = HERE.parent / "docs" / "guide"

# <!-- check: ch25.evidence_swing_nats = 191.1 ± 0.05 -->   (± or +/-)
TAG = re.compile(
    r"<!--\s*check:\s*(?P<ch>\w+)\.(?P<key>\w+)\s*=\s*"
    r"(?P<val>-?[\d.eE+-]+)\s*(?:±|\+/-)\s*(?P<tol>[\d.eE+-]+)\s*-->"
)


def main() -> int:
    values = {k: spec["fn"]() for k, spec in W._EX.items()}

    n_ok = n_bad = n_miss = 0
    problems: list[str] = []

    for path in sorted(GUIDE.glob("*.md")):
        text = path.read_text()
        for m in TAG.finditer(text):
            ch, key = m.group("ch"), m.group("key")
            line = text.count("\n", 0, m.start()) + 1
            where = f"{path.name}:{line}"

            if ch not in values:
                problems.append(f"{where}  no example '{ch}' in worked_examples.py")
                n_miss += 1
                continue
            if key not in values[ch]:
                problems.append(f"{where}  '{ch}' has no key '{key}'")
                n_miss += 1
                continue

            got = float(values[ch][key])
            want, tol = float(m.group("val")), float(m.group("tol"))
            if abs(got - want) > tol:
                problems.append(
                    f"{where}  {ch}.{key}: prose says {want} ± {tol}, "
                    f"code computes {got:.6g}  (off by {abs(got - want):.3g})")
                n_bad += 1
            else:
                n_ok += 1

    for p in problems:
        print("  " + p)

    total = n_ok + n_bad + n_miss
    print(f"\n{n_ok}/{total} tagged numbers reproduce", end="")
    if n_bad:
        print(f"  |  {n_bad} WRONG", end="")
    if n_miss:
        print(f"  |  {n_miss} unresolvable", end="")
    print()
    return 1 if (n_bad or n_miss) else 0


if __name__ == "__main__":
    raise SystemExit(main())
