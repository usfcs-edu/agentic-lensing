"""The guides this toolchain builds. One dict, one source of truth.

Every tool (`make_figures`, `make_examples`, `verify_numbers`, `lint_guide`,
`build_guide`) takes `--guide <name>` and derives all of its paths from here, so
adding a third book is a dict entry rather than a fork.

Namespacing is PHYSICAL, not by string prefix: each guide gets its own `figures`
and `worked_examples` module, and a driver imports exactly one per process. That
makes a collision structurally impossible instead of merely defended against —
both guides can have a `ch05` and neither can see the other's.

Title/subtitle deliberately are NOT here: they live in each guide's
`contract/outline.yml` under `meta:`, which is where the authoring agents read
them from. Duplicating them into this dict is how they drift.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

GUIDES: dict[str, dict] = {
    "guide": dict(
        docs="guide",                       # -> site/docs/guide/
        figures="figures",                  # importable module (guide_src/)
        examples="worked_examples",
        contract="contract",                # -> guide_src/contract/
        manifest="figures.json",
        examples_json="worked_examples.json",
        out="guide",                        # -> repo/guide/guide.{pdf,html}
    ),
    "primer": dict(
        docs="primer",                      # -> site/docs/primer/
        figures="primer.figures",           # -> guide_src/primer/figures.py
        examples="primer.worked_examples",
        contract="primer/contract",
        manifest="primer/figures.json",
        examples_json="primer/worked_examples.json",
        out="primer",                       # -> repo/guide/primer.{pdf,html}
    ),
}

DEFAULT = "guide"


def spec(name: str) -> dict:
    if name not in GUIDES:
        raise SystemExit(
            f"unknown guide {name!r} — known: {', '.join(sorted(GUIDES))}")
    g = dict(GUIDES[name])
    g["name"] = name
    g["docs_dir"] = REPO / "site" / "docs" / g["docs"]
    g["fig_dir"] = g["docs_dir"] / "figures"
    g["manifest_path"] = HERE / g["manifest"]
    g["examples_path"] = HERE / g["examples_json"]
    g["contract_dir"] = HERE / g["contract"]
    g["out_stem"] = REPO / "guide" / g["out"]
    return g


def meta(name: str) -> dict:
    """Title/subtitle from the guide's own outline.yml `meta:` block.

    Deliberately a 6-line parser rather than a PyYAML dependency: the mkdocs
    venv has yaml but the figure venv need not, and this reads two scalars.
    """
    f = spec(name)["contract_dir"] / "outline.yml"
    out, in_meta = {}, False
    for line in f.read_text().split("\n"):
        if line.startswith("meta:"):
            in_meta = True
            continue
        if in_meta:
            if line and not line[0].isspace():
                break
            m = line.strip()
            for key in ("title", "subtitle"):
                if m.startswith(f"{key}:"):
                    out[key] = m.split(":", 1)[1].strip().strip('"').strip("'")
    return out


def add_argument(ap) -> None:
    """Shared `--guide` flag, so every tool spells it the same way."""
    ap.add_argument("--guide", default=DEFAULT, choices=sorted(GUIDES),
                    help=f"which guide to operate on (default: {DEFAULT})")
