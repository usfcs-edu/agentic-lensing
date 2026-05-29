#!/usr/bin/env python3
"""
exp_analyze.py — analyze and compare exp_run.py runs.

Usage:
    exp_analyze.py --digest experiments/runs/<run_id>
    exp_analyze.py --compare experiments/runs/<run_a> experiments/runs/<run_b> ...
                   [--out comparison.md] [--plot comparison.png]

--digest emits a single-paragraph natural-language summary suitable for pasting
into RESEARCH_LOG.md (peak metric, time-to-best, convergence judgement).

--compare emits a markdown table comparing the runs side-by-side and (if
matplotlib finds a usable backend) a PNG of val curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict[str, Any]:
    spec = yaml.safe_load((run_dir / "spec.yaml").read_text())
    metrics_path = run_dir / "metrics.jsonl"
    records: list[dict] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    summary_md = (run_dir / "summary.md").read_text() if (run_dir / "summary.md").exists() else ""
    return {
        "dir": run_dir,
        "spec": spec,
        "records": records,
        "summary_md": summary_md,
    }


# ---------------------------------------------------------------------------
# Digest (single-run, paragraph)
# ---------------------------------------------------------------------------

def digest(run: dict) -> str:
    spec = run["spec"]
    recs = run["records"]
    name = spec.get("name", run["dir"].name)
    repo = spec.get("repo", "?")
    desc = spec.get("description", "")

    if not recs:
        return f"**{name}** ({repo}): produced no parseable metrics. Check `{run['dir']}/stdout.log`."

    if repo == "codecs":
        return _digest_codecs(name, desc, recs)
    if repo == "redshifty":
        return _digest_redshifty(name, desc, recs)
    return f"**{name}** ({repo}): {len(recs)} records (no repo-specific digest)."


def _digest_codecs(name: str, desc: str, recs: list[dict]) -> str:
    steps = [r["step"] for r in recs if "step" in r]
    first, last = recs[0], recs[-1]
    n = len(recs)

    def grab(k):
        return [r.get(k) for r in recs if k in r]

    train = grab("train_loss")
    val_nll = grab("val_nll")
    val_r2 = grab("val_r2")
    perp = grab("val_perplexity")

    parts = [
        f"**{name}** (codecs): {n} evaluations across steps {steps[0]}–{steps[-1]}."
    ]
    if desc:
        parts[-1] += f" {desc}."
    if train:
        parts.append(f"`train_loss` went from {train[0]:.3g} to {train[-1]:.3g} (min {min(train):.3g}).")
    if val_nll:
        parts.append(f"`val_nll` went from {val_nll[0]:.3g} to {val_nll[-1]:.3g} (min {min(val_nll):.3g}).")
    if val_r2:
        parts.append(f"`val_r2` reached {max(val_r2):.3g} (final {val_r2[-1]:.3g}).")
    if perp:
        parts.append(f"RFSQ `val_perplexity` final {perp[-1]:.3g}.")

    # quick convergence judgement
    if val_nll and len(val_nll) >= 3:
        if val_nll[-1] < val_nll[0] * 0.5:
            parts.append("Loss is dropping clearly — training is learning.")
        elif val_nll[-1] < val_nll[0] * 0.95:
            parts.append("Loss is descending mildly — needs more steps.")
        else:
            parts.append("Loss is essentially flat — investigate.")

    return " ".join(parts)


def _digest_redshifty(name: str, desc: str, recs: list[dict]) -> str:
    by_ap: dict[str, list[dict]] = {}
    for r in recs:
        by_ap.setdefault(r.get("approach", "?"), []).append(r)
    parts: list[str] = [f"**{name}** (redshifty): {len(recs)} epoch records across approaches {sorted(by_ap)}."]
    if desc:
        parts[-1] += f" {desc}."
    for ap, ar in sorted(by_ap.items()):
        train = [r.get("train_loss") for r in ar if "train_loss" in r]
        val = [r.get("val_loss") for r in ar if "val_loss" in r]
        zacc = [r.get("val_redshift_acc") for r in ar if "val_redshift_acc" in r]
        seg = [f"Approach {ap.upper()}: {len(ar)} epochs"]
        if train:
            seg.append(f"train_loss {train[0]:.3g}→{train[-1]:.3g}")
        if val:
            seg.append(f"val_loss {val[0]:.3g}→{val[-1]:.3g} (min {min(val):.3g})")
        if zacc:
            seg.append(f"val_z_acc max {max(zacc):.3g}")
        parts.append("; ".join(seg) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Compare (multi-run, markdown + PNG)
# ---------------------------------------------------------------------------

def compare_md(runs: list[dict]) -> str:
    cols = ["run", "repo", "name", "records", "first_metric", "final_metric", "best_metric", "notes"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for run in runs:
        spec = run["spec"]
        recs = run["records"]
        row = {
            "run": run["dir"].name,
            "repo": spec.get("repo", "?"),
            "name": spec.get("name", ""),
            "records": str(len(recs)),
            "first_metric": "",
            "final_metric": "",
            "best_metric": "",
            "notes": spec.get("description", "")[:80],
        }
        if recs:
            key, label = _headline_metric(spec.get("repo"), recs)
            vals = [r[key] for r in recs if key in r]
            if vals:
                row["first_metric"] = f"{label}={vals[0]:.4g}"
                row["final_metric"] = f"{label}={vals[-1]:.4g}"
                loss_like = any(s in key for s in ("loss", "nll", "total", "recon", "bce"))
                if loss_like:
                    row["best_metric"] = f"{label}_min={min(vals):.4g}"
                else:
                    row["best_metric"] = f"{label}_max={max(vals):.4g}"
        lines.append("| " + " | ".join(row[c] for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _headline_metric(repo: str | None, records: list[dict] | None = None) -> tuple[str, str]:
    # Try the fallback chain against actual records to pick a key that exists.
    chains: dict[str | None, list[str]] = {
        "codecs": ["val_nll", "val_loss"],
        "redshifty": ["val_total", "val_loss", "val_recon", "val_redshift_acc"],
    }
    chain = chains.get(repo, ["val_loss"])
    if records:
        for k in chain:
            if any(k in r for r in records):
                return k, k
    return chain[0], chain[0]


def compare_plot(runs: list[dict], out_path: Path) -> bool:
    """Plot val curves for each run on a shared axes. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}", file=sys.stderr)
        return False

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    any_plotted = False
    for run in runs:
        spec = run["spec"]
        recs = run["records"]
        key, label = _headline_metric(spec.get("repo"), recs)
        # x-axis: prefer 'step', fall back to 'epoch'
        xkey = "step" if any("step" in r for r in recs) else "epoch"
        xs = [r[xkey] for r in recs if xkey in r and key in r]
        ys = [r[key]  for r in recs if xkey in r and key in r]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o",
                label=f"{run['dir'].name} ({spec.get('repo','?')}/{label}, x={xkey})")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("step or epoch")
    ax.set_ylabel("headline metric")
    ax.set_title("exp_analyze: val curve comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--digest", type=Path, default=None, help="Path to a single run directory")
    grp.add_argument("--compare", nargs="+", default=None, help="Two or more run directories")
    ap.add_argument("--out", type=Path, default=None, help="Markdown output path (compare mode)")
    ap.add_argument("--plot", type=Path, default=None, help="PNG output path (compare mode)")
    args = ap.parse_args()

    if args.digest:
        run = load_run(args.digest)
        print(digest(run))
        return 0

    runs = [load_run(Path(p)) for p in args.compare]
    md = compare_md(runs)
    if args.out:
        args.out.write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)

    if args.plot:
        if compare_plot(runs, args.plot):
            print(f"wrote {args.plot}")
        else:
            print("no plot written (no parseable metric series)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
