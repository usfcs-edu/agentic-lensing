#!/usr/bin/env python3
"""verify_against_reference.py — compare the Apple Silicon (MPS) reproduction
outputs against the committed phoenix reference numbers, and write a
PASS/FAIL comparison report to data/REPRODUCTION_MPS_COMPARE.md.

Pure-Python (pandas only). Run after training and/or recovery have produced
outputs under apple-silicon/data/. Reference artifacts live in
apple-silicon/data/ref/ (copied from the committed ../data and the phoenix
transfer). Sections whose inputs are not present yet are reported as PENDING
and do not fail the run.

Tolerances (see plan §7): test AUC gated at an absolute floor; recovery at
p>=0.9 gated on |MPS - reference| with per-grade tolerances wide enough to
absorb training stochasticity (the dominant source of variance, not the MPS
backend); candidate-pool sizes are informational; the optional same-checkpoint
MPS-vs-phoenix score cross-check is gated tightly (pure float drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REF = DATA / "ref"

AUC_FLOOR = 0.994                                  # absolute gate on test AUC
REC_TOL = {"A": 0.02, "B": 0.02, "C": 0.03, "ALL": 0.02}   # |MPS-ref| at p>=0.9
POOL_TOL_FRAC = 0.10                               # candidate-pool relative tol (informational)
XCHECK_TOL = 1e-3                                  # same-ckpt MPS vs phoenix scores

rows: list[dict] = []
overall_ok = True


def rec(section, metric, reference, mps, tol="", ok=None, gate=True):
    global overall_ok
    if ok is False and gate:
        overall_ok = False
    rows.append({"section": section, "metric": metric, "reference": reference,
                 "mps": mps, "tol": tol, "ok": ok})


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---- 1. Training: held-out test AUC ----
for tag, fn in (("DR9", "test_result.json"), ("DR7", "test_result_dr7.json")):
    mps = load_json(DATA / fn)
    ref = load_json(REF / fn)
    ref_auc = round(float(ref["test_auc"]), 4) if ref and "test_auc" in ref else None
    if mps is None or "test_auc" not in mps:
        rec("training", f"{tag} test_auc", ref_auc, "PENDING", f">={AUC_FLOOR}", None)
        continue
    auc = round(float(mps["test_auc"]), 4)
    ok = auc >= AUC_FLOOR
    rec("training", f"{tag} test_auc", ref_auc, auc, f">={AUC_FLOOR}", ok)
    if ref and "best_val_auc" in mps:
        rec("training", f"{tag} best_val_auc (info)",
            round(float(ref.get("best_val_auc", float('nan'))), 4),
            round(float(mps["best_val_auc"]), 4), "info", None, gate=False)


# ---- 2. Recovery table at p>=0.9 ----
mps_cmp, ref_cmp = DATA / "recovery_compare.csv", REF / "recovery_compare.csv"
if mps_cmp.exists() and ref_cmp.exists():
    m = pd.read_csv(mps_cmp)
    r = pd.read_csv(ref_cmp)
    j = m.merge(r, on=["grade", "threshold"], suffixes=("_mps", "_ref"))
    j = j[abs(j["threshold"] - 0.9) < 1e-9]
    order = {"A": 0, "B": 1, "C": 2, "ALL": 3}
    j = j.sort_values("grade", key=lambda s: s.map(order))
    for _, row in j.iterrows():
        g = row["grade"]
        tol = REC_TOL[g]
        for col in ("frac_DR9-trained", "frac_DR7-trained"):
            mv, rv = float(row[f"{col}_mps"]), float(row[f"{col}_ref"])
            ok = abs(mv - rv) <= tol
            rec("recovery@p>=0.9", f"{g}  {col.replace('frac_', '')}",
                f"{rv:.3f}", f"{mv:.3f}", f"+/-{tol:.2f}", ok)
else:
    rec("recovery@p>=0.9", "recovery_compare.csv",
        "present" if ref_cmp.exists() else "no-ref",
        "PENDING" if not mps_cmp.exists() else "present", "", None)


# ---- 3. Candidate-pool sizes (informational) ----
for tag, fn, ref_n in (("DR9", "inference_scores_dr9trained.parquet", 74011),
                       ("DR7", "inference_scores_dr7trained.parquet", 25792)):
    p = DATA / fn
    if p.exists():
        n = int((pd.read_parquet(p, columns=["score"])["score"] >= 0.9).sum())
        ok = abs(n - ref_n) <= POOL_TOL_FRAC * ref_n
        rec("pool@p>=0.9 (info)", f"{tag} n(score>=0.9)", f"{ref_n:,}", f"{n:,}",
            f"+/-{int(POOL_TOL_FRAC*100)}%", ok, gate=False)


# ---- 4. MPS inference fidelity cross-check (same checkpoint, optional) ----
x = load_json(DATA / "mps_xcheck.json")
if x and "max_abs_delta" in x:
    mae = float(x["max_abs_delta"])
    ok = mae <= XCHECK_TOL
    rec("mps-fidelity", f"max|score_mps - score_phoenix| (n={x.get('n', '?')})",
        f"<= {XCHECK_TOL}", f"{mae:.2e}", f"<= {XCHECK_TOL}", ok)


# ---- write report ----
def cell(v):
    return "" if v is None else str(v)


def mark(ok):
    return {True: "PASS", False: "FAIL", None: "PENDING"}[ok]


prov = ""
try:
    import torch  # noqa
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    prov = f"torch {torch.__version__}, device={dev}"
except Exception:
    prov = "torch unavailable"

lines = [
    "# Apple Silicon (MPS) reproduction vs. phoenix reference",
    "",
    f"Provenance: {prov}",
    "",
    "| section | metric | reference | MPS | tol | result |",
    "| :--- | :--- | ---: | ---: | :--- | :--- |",
]
for r in rows:
    lines.append(f"| {r['section']} | {r['metric']} | {cell(r['reference'])} | "
                 f"{cell(r['mps'])} | {r['tol']} | {mark(r['ok'])} |")
gated = [r for r in rows if r["ok"] is not None and r["tol"] != "info"]
n_pass = sum(1 for r in gated if r["ok"])
n_pend = sum(1 for r in rows if r["ok"] is None)
lines += [
    "",
    f"**Gated checks:** {n_pass}/{len(gated)} passed"
    f"{f'  ({n_pend} pending)' if n_pend else ''}.",
    "",
    f"## {'OVERALL: PASS' if overall_ok else 'OVERALL: FAIL'}",
]
out = DATA / "REPRODUCTION_MPS_COMPARE.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[done] wrote {out}")
raise SystemExit(0 if overall_ok else 1)
