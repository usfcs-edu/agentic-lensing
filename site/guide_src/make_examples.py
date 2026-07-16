#!/usr/bin/env python
"""Driver for a guide's worked examples.

    ~/.venvs/lensjudge/bin/python site/guide_src/make_examples.py --guide guide --check
    ~/.venvs/lensjudge/bin/python site/guide_src/make_examples.py --guide primer --emit
    ~/.venvs/lensjudge/bin/python site/guide_src/make_examples.py --guide guide --show ch25

Separate from `worked_examples.py` because that module IS the content: it
self-registers on import, so it cannot also be the thing that chooses which
content module to import. (`registry.py`'s docstring records the first time this
codebase learned that lesson; this is the second.)

`--check` asserts every value pinned against a number this repository has
published. `--emit` writes the JSON the prose's `<!-- check: -->` tags are
verified against by `verify_numbers.py`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

import examples_registry as R
import guides


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    guides.add_argument(ap)
    ap.add_argument("--check", action="store_true", help="assert pinned values")
    ap.add_argument("--emit", action="store_true", help="write the JSON")
    ap.add_argument("--show", metavar="KEY", help="print one example's values")
    args = ap.parse_args()

    g = guides.spec(args.guide)
    importlib.import_module(g["examples"])   # registers, exactly once

    if args.show:
        if args.show not in R._EX:
            print(f"no example {args.show!r} in {args.guide}; have: "
                  f"{', '.join(sorted(R._EX))}", file=sys.stderr)
            return 2
        print(json.dumps(R._EX[args.show]["fn"](), indent=2, default=float))
        return 0

    if args.check:
        n, fails = R.check()
        if fails:
            print(f"FAIL — {args.guide}: worked examples do not reproduce:")
            for f in fails:
                print("  " + f)
            return 1
        print(f"OK — {n} pinned values across {len(R._EX)} examples reproduce "
              f"({args.guide})")
        return 0

    if args.emit:
        out = g["examples_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(R.run_all(), indent=2, sort_keys=True, default=float) + "\n")
        print(f"-> {out}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
