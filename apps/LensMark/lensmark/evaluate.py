"""Evaluation: join critiques with their ProposalRun and summarise by model / effort.

Per critiqued item the verdict maps to a category: ``correct`` -> TP; ``wrong_position | wrong_size |
wrong_label | wrong_type`` -> PARTIAL; ``spurious | redundant`` -> FP; ``missed_by_model`` -> FN.
``eval_rows`` groups the per-run rows by the ``--by`` columns and writes
``exports/eval/items.parquet`` (one row per critiqued item, all join columns), ``exports/eval/per_run.csv``
(one row per critique) and ``exports/eval/runs.csv`` (the grouped table it returns). Columns are named
like lensjudge's regrade tables (model, effort, ...) so the two can be joined.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Optional

from .critique import find_run, list_critiques
from .model import Critique, LensMarkFile, MaskCircle, ProposalRun
from .store import Campaign

CATEGORY: dict[str, str] = {
    "correct": "TP",
    "wrong_position": "PARTIAL", "wrong_size": "PARTIAL", "wrong_label": "PARTIAL", "wrong_type": "PARTIAL",
    "spurious": "FP", "redundant": "FP",
    "missed_by_model": "FN",
}
RUN_COLUMNS = ("image_id", "run_id", "reviewer", "reviewed_at", "model", "effort", "engine", "cost_usd",
               "duration_s", "n_items_proposed", "n_invalid", "n_repaired", "parse_ok", "lead_time_s",
               "proposed", "proposed_masks", "theta_e_human_arcsec", "theta_e_proposed_arcsec", "theta_e_abs_err",
               "completeness", "geometric_accuracy", "label_quality", "description_quality", "theta_e_verdict",
               "would_use_as_fewshot")
ITEM_COLUMNS = RUN_COLUMNS + ("item_id", "item_type", "kind", "status", "created_by_kind", "verdict", "severity",
                              "category", "delta_arcsec", "comment")
GROUP_METRICS = ("n_runs", "n_images", "proposed", "TP", "PARTIAL", "FP", "FN", "precision_strict",
                 "precision_lenient", "recall", "spurious_mask_rate", "median_delta_arcsec", "theta_e_abs_err",
                 "mean_cost_usd", "parse_ok_rate", "mean_n_invalid", "mean_lead_time_s")


# ----------------------------------------------------------------------------- joins
def _run_from_proposal_file(campaign: Campaign, critique: Critique) -> Optional[ProposalRun]:
    """Fallback when the file's provenance lost the run: read proposals/<id>.<run_id>.json leniently."""
    p = campaign.proposals_dir / f"{critique.image_id}.{critique.run_id}.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    cand = d.get("run") if isinstance(d.get("run"), dict) else d
    cand = {k: v for k, v in cand.items() if k in ProposalRun.model_fields}
    cand.setdefault("run_id", critique.run_id)
    cand.setdefault("model", critique.model or "unknown")
    try:
        return ProposalRun.model_validate(cand)
    except ValueError:
        return None


def _proposed_theta_e(run: Optional[ProposalRun]) -> Optional[float]:
    if run is None or not isinstance(run.proposed_system, dict):
        return None
    te = run.proposed_system.get("theta_e")
    if isinstance(te, dict):
        v = te.get("value_arcsec")
        return float(v) if isinstance(v, (int, float)) else None
    return float(te) if isinstance(te, (int, float)) else None


def run_columns(campaign: Campaign, critique: Critique, file: Optional[LensMarkFile],
                run: Optional[ProposalRun]) -> dict[str, Any]:
    """The join columns shared by every item row of one critique."""
    proposed_masks = 0
    if file is not None:
        proposed_masks = sum(1 for it in file.items if isinstance(it, MaskCircle)
                             and it.created_by.kind == "claude" and it.created_by.run_id == critique.run_id)
    proposed = critique.counts.get("proposed")
    if proposed is None:
        proposed = run.n_items_proposed if run is not None else sum(
            1 for ci in critique.items if CATEGORY[ci.verdict] != "FN")
    te_h = critique.panel.theta_e_human_arcsec
    te_p = _proposed_theta_e(run)
    return {
        "image_id": critique.image_id, "run_id": critique.run_id, "reviewer": critique.reviewer,
        "reviewed_at": critique.reviewed_at,
        "model": (run.model if run else None) or critique.model,
        "effort": (run.effort if run else None) or critique.effort,
        "engine": run.engine if run else None,
        "cost_usd": run.cost_usd if run else None, "duration_s": run.duration_s if run else None,
        "n_items_proposed": run.n_items_proposed if run else None,
        "n_invalid": run.n_invalid if run else None, "n_repaired": run.n_repaired if run else None,
        "parse_ok": run.parse_ok if run else None, "lead_time_s": critique.lead_time_s,
        "proposed": int(proposed), "proposed_masks": proposed_masks,
        "theta_e_human_arcsec": te_h, "theta_e_proposed_arcsec": te_p,
        "theta_e_abs_err": abs(te_h - te_p) if (te_h is not None and te_p is not None) else None,
        "completeness": critique.panel.completeness, "geometric_accuracy": critique.panel.geometric_accuracy,
        "label_quality": critique.panel.label_quality, "description_quality": critique.panel.description_quality,
        "theta_e_verdict": critique.panel.theta_e_verdict, "would_use_as_fewshot": critique.panel.would_use_as_fewshot,
    }


def item_rows(campaign: Campaign) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(one row per critiqued item, one row per critique with TP/PARTIAL/FP/FN/spurious_masks tallied)."""
    items: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    cache: dict[str, Optional[LensMarkFile]] = {}
    for c in list_critiques(campaign):
        if c.image_id not in cache:
            cache[c.image_id] = campaign.load(c.image_id)
        file = cache[c.image_id]
        run = find_run(file, c.run_id) or _run_from_proposal_file(campaign, c)
        base = run_columns(campaign, c, file, run)
        tally = {"TP": 0, "PARTIAL": 0, "FP": 0, "FN": 0, "spurious_masks": 0, "deltas": []}
        for ci in c.items:
            it = file.item(ci.item_id) if file is not None else None
            cat = CATEGORY[ci.verdict]
            tally[cat] += 1
            kind = getattr(it, "kind", None) if it is not None else None
            if ci.verdict == "spurious" and (isinstance(it, MaskCircle) or (it is None and ci.item_id.startswith("ann-mask-"))):
                tally["spurious_masks"] += 1
            delta = ci.delta_arcsec
            if delta is None and it is not None and it.review is not None:
                delta = it.review.delta_arcsec
            if delta is not None:
                tally["deltas"].append(float(delta))
            items.append({**base, "item_id": ci.item_id, "item_type": it.type if it is not None else None,
                          "kind": kind, "status": it.status if it is not None else None,
                          "created_by_kind": it.created_by.kind if it is not None else None,
                          "verdict": ci.verdict, "severity": ci.severity, "category": cat,
                          "delta_arcsec": delta, "comment": ci.comment})
        runs.append({**base, **{k: tally[k] for k in ("TP", "PARTIAL", "FP", "FN", "spurious_masks")},
                     "deltas": tally["deltas"]})
    return items, runs


# ----------------------------------------------------------------------------- grouping
def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _mean(vals: list[Any]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def group_rows(runs: list[dict[str, Any]], by: list[str]) -> list[dict[str, Any]]:
    if runs:
        missing = [b for b in by if b not in runs[0]]
        if missing:
            raise ValueError(f"unknown --by column(s) {missing}; choose from {sorted(RUN_COLUMNS)}")
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault(tuple(r.get(b) for b in by), []).append(r)
    out: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        rs = groups[key]
        tp = sum(r["TP"] for r in rs)
        partial = sum(r["PARTIAL"] for r in rs)
        fp = sum(r["FP"] for r in rs)
        fn = sum(r["FN"] for r in rs)
        proposed = sum(r["proposed"] for r in rs)
        masks = sum(r["proposed_masks"] for r in rs)
        deltas = [d for r in rs for d in r["deltas"]]
        parse_flags = [r["parse_ok"] for r in rs if r["parse_ok"] is not None]
        row: dict[str, Any] = {b: key[i] for i, b in enumerate(by)}
        row.update({
            "n_runs": len({(r["image_id"], r["run_id"]) for r in rs}),
            "n_images": len({r["image_id"] for r in rs}),
            "proposed": proposed, "TP": tp, "PARTIAL": partial, "FP": fp, "FN": fn,
            "precision_strict": _ratio(tp, proposed),
            "precision_lenient": _ratio(tp + partial, proposed),
            "recall": _ratio(tp + partial, tp + partial + fn),
            "spurious_mask_rate": _ratio(sum(r["spurious_masks"] for r in rs), masks),
            "median_delta_arcsec": statistics.median(deltas) if deltas else None,
            "theta_e_abs_err": _mean([r["theta_e_abs_err"] for r in rs]),
            "mean_cost_usd": _mean([r["cost_usd"] for r in rs]),
            "parse_ok_rate": _ratio(sum(1 for p in parse_flags if p), len(parse_flags)),
            "mean_n_invalid": _mean([r["n_invalid"] for r in rs]),
            "mean_lead_time_s": _mean([r["lead_time_s"] for r in rs]),
        })
        out.append(row)
    return out


def eval_rows(campaign: Campaign, by: str = "model,effort", *, out: str | Path | None = None,
              write: bool = True) -> list[dict[str, Any]]:
    """Grouped evaluation rows; also writes exports/eval/{items.parquet, per_run.csv, runs.csv} unless ``write=False``."""
    import pandas as pd

    by_cols = [b.strip() for b in by.split(",") if b.strip()] or ["model", "effort"]
    items, runs = item_rows(campaign)
    rows = group_rows(runs, by_cols)
    if write:
        out_dir = Path(out).expanduser() if out else campaign.exports_dir / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(items, columns=list(ITEM_COLUMNS)).to_parquet(out_dir / "items.parquet", index=False)
        per_run = [{k: v for k, v in r.items() if k != "deltas"} for r in runs]
        pd.DataFrame(per_run, columns=list(RUN_COLUMNS) + ["TP", "PARTIAL", "FP", "FN", "spurious_masks"]).to_csv(
            out_dir / "per_run.csv", index=False)
        pd.DataFrame(rows, columns=by_cols + list(GROUP_METRICS)).to_csv(out_dir / "runs.csv", index=False)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no critiques yet)"
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def cli_eval(dir: str, *, by: str = "model,effort", out: Optional[str] = None) -> int:
    campaign = Campaign(dir)
    rows = eval_rows(campaign, by, out=out)
    print(format_table(rows))
    out_dir = Path(out).expanduser() if out else campaign.exports_dir / "eval"
    print(f"\nwrote {out_dir / 'items.parquet'}, {out_dir / 'per_run.csv'}, {out_dir / 'runs.csv'}")
    return 0
