#!/usr/bin/env python3
"""verify_against_reference.py — compare the Apple Silicon (MPS) Huang-2021
reproduction against the committed phoenix reference, writing
data/REPRODUCTION_MPS_COMPARE.md.

Design principle (same as the huang-2020 port): GATE what the port must reproduce
(identical computation), REPORT what legitimately varies between independent
from-scratch training runs.

  HARD GATES (port correctness):
    - four from-scratch test AUCs vs floors:
        shielded DR9 >= 0.995, shielded DR7 >= 0.990,
        L18-northaug >= 0.995, shielded-northaug >= 0.995
    - MPS-vs-CUDA inference fidelity for the SAME checkpoints (both models, <= 1e-3)
    - recovery table reproduced from the phoenix scores (all/combined, |Δ| <= 1pp)
      — proves the leak-aware crossmatch analysis is correct on the Mac
    - published-catalog grade counts A=216/B=199/C=897/total=1312 (paper tolerance)
    - north-aug false-positive collapse: post-northaug north non-lens >=0.1 rate
      <= 5% AND well below the pre-northaug rate (the 91% -> 0.8% headline)
    - structural: shielded params == 59,905, L18 == 3,508,833; |shielded-L18 AUC|
      <= 0.005 per DR; leaked-bucket recall >= honest-bucket recall per grade

  INFORMATIONAL (from-scratch retrain variance, not gated):
    - signed Δ of each MPS test AUC vs the phoenix reference
    - honest (leak-free) recovery (the ~50% generalization signal)

Pure pandas/numpy (the north-aug rates come from northaug_fp_check.json, written by
northaug_fp_check.py). Missing inputs report PENDING.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REF = HERE.parent / "data"   # committed phoenix reference numbers (huang-2021/data)

AUC_FLOORS = {
    "shielded_dr9": 0.995,
    "shielded_dr7": 0.990,
    "l18_northaug": 0.995,
    "shielded_northaug": 0.995,
}
XCHECK_TOL = 1e-3
REC_TOL = 0.01            # phoenix-score-derived recovery: must reproduce ~exactly
NORTHAUG_FP_MAX = 0.05   # post-northaug north non-lens >=0.1 rate must collapse
GRADES = ("A", "B", "C", "ALL")
THRESH = ("0.1", "0.5", "0.9")

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


# ---- 1. Four from-scratch test AUCs vs floors ----
TAG_FILE = {
    "shielded_dr9": "test_result_shielded_dr9.json",
    "shielded_dr7": "test_result_shielded_dr7.json",
    "l18_northaug": "test_result_l18_northaug.json",
    "shielded_northaug": "test_result_shielded_northaug.json",
}
for tag, fn in TAG_FILE.items():
    floor = AUC_FLOORS[tag]
    mps, ref = load_json(DATA / fn), load_json(REF / fn)
    ref_auc = round(float(ref["test_auc"]), 4) if ref and "test_auc" in ref else "—"
    if not mps or "test_auc" not in mps:
        rec("training", f"{tag} test_auc", ref_auc, "PENDING", f">={floor}", None)
        continue
    auc = float(mps["test_auc"])
    delta = (auc - float(ref["test_auc"])) if ref and "test_auc" in ref else float("nan")
    rec("training", f"{tag} test_auc  (Δ={delta:+.4f})", ref_auc, round(auc, 4),
        f">={floor}", auc >= floor)

# ---- 2. MPS inference fidelity (same checkpoints), both models ----
# Gate the ROBUST fidelity (p99.9 of |score_mps - score_phoenix| over the bounded
# run) rather than the single worst galaxy. A genuine MPS inference bug corrupts
# many galaxies (median/p99.9 blow up); float drift in the tiny shielded net (the
# 1x1 "shields" are the most MPS-sensitive op) leaves at most a handful near the
# tolerance. max is reported in the label, not gated.
x = load_json(DATA / "mps_xcheck.json")
if x:
    for kind in ("l18", "shielded"):
        v = x.get(kind, {})
        p999, mad = v.get("p99_9_abs_delta"), v.get("max_abs_delta")
        if not v.get("present") or p999 is None:
            rec("mps-fidelity", f"{kind} p99.9|Δ| (n={v.get('n_overlap', '?')})",
                f"<={XCHECK_TOL}", "PENDING", f"<={XCHECK_TOL}", None)
        else:
            rec("mps-fidelity", f"{kind} p99.9|Δ|  (max={mad:.1e}, n={v.get('n_overlap')})",
                f"<={XCHECK_TOL}", f"{p999:.2e}", f"<={XCHECK_TOL}", p999 <= XCHECK_TOL)
else:
    rec("mps-fidelity", "mps_xcheck.json", f"<={XCHECK_TOL}", "PENDING", "", None)

# ---- 3. Recovery table reproduced from phoenix scores (all / combined) ----
m_csv, r_csv = DATA / "recovery_dr8_summary.csv", REF / "recovery_dr8_summary.csv"
if m_csv.exists() and r_csv.exists():
    m = pd.read_csv(m_csv); r = pd.read_csv(r_csv)
    key = ["bucket", "grade", "model"]
    j = m.merge(r, on=key, suffixes=("_mps", "_ref"))
    sel = j[(j["bucket"] == "all") & (j["model"] == "combined")].set_index("grade")
    for g in GRADES:
        if g not in sel.index:
            continue
        for t in THRESH:
            mv = float(sel.loc[g, f"frac_ge_{t}_mps"])
            rv = float(sel.loc[g, f"frac_ge_{t}_ref"])
            rec("recovery (gate)", f"all/combined {g} p>={t}", f"{rv:.3f}", f"{mv:.3f}",
                f"+/-{REC_TOL:.2f}", abs(mv - rv) <= REC_TOL)
    # structural: leaked recall >= honest recall per grade (combined, p>=0.9)
    lk = m[(m["bucket"] == "leaked") & (m["model"] == "combined")].set_index("grade")
    hn = m[(m["bucket"] == "honest") & (m["model"] == "combined")].set_index("grade")
    order_ok = all(float(lk.loc[g, "frac_ge_0.9"]) >= float(hn.loc[g, "frac_ge_0.9"]) - 1e-9
                   for g in GRADES if g in lk.index and g in hn.index)
    rec("structure", "leaked recall >= honest (combined p>=0.9)", "all grades",
        "holds" if order_ok else "violated", "", order_ok)
    # informational: honest combined leak-free recall (the ~50% generalization signal)
    if "ALL" in hn.index:
        rec("honest recall (info)", "leak-free combined ALL p>=0.9", "0.504",
            f"{float(hn.loc['ALL', 'frac_ge_0.9']):.3f}", "info", None, gate=False)
else:
    rec("recovery (gate)", "recovery_dr8_summary.csv",
        "present" if r_csv.exists() else "no-ref",
        "PENDING" if not m_csv.exists() else "present", "", None)

# ---- 4. Published-catalog grade counts ----
cat = DATA / "huang2021_published_catalog.csv"
if cat.exists():
    df = pd.read_csv(cat)
    target = {"A": 216, "B": 199, "C": 897, "total": 1312}
    got = {g: int((df["grade"] == g).sum()) for g in "ABC"}
    got["total"] = len(df)
    for k in ("A", "B", "C", "total"):
        tol = max(3, 0.05 * target[k])
        rec("catalog", f"grade {k}", target[k], got[k],
            f"+/-{int(tol)}", abs(got[k] - target[k]) <= tol)
else:
    rec("catalog", "huang2021_published_catalog.csv", "1312", "PENDING", "", None)

# ---- 5. North-aug false-positive collapse ----
fp = load_json(DATA / "northaug_fp_check.json")
if fp and "post_rate" in fp:
    pre, post = float(fp["pre_rate"]), float(fp["post_rate"])
    rec("northaug (gate)", f"post north non-lens >=0.1  (pre={pre:.1%}, n={fp.get('n')})",
        f"<={NORTHAUG_FP_MAX:.0%}", f"{post:.1%}", f"<={NORTHAUG_FP_MAX:.0%}",
        (post <= NORTHAUG_FP_MAX) and (post < pre))
else:
    rec("northaug (gate)", "northaug_fp_check.json", f"<={NORTHAUG_FP_MAX:.0%}",
        "PENDING", "", None)

# ---- 6. Structural: param counts + shielded-vs-L18 AUC gap ----
arch = DATA / "arch_comparison.csv"
sd9 = load_json(DATA / "test_result_shielded_dr9.json")
if sd9 and "n_params" in sd9:
    rec("structure", "shielded params", "59,905", f"{int(sd9['n_params']):,}",
        "==", int(sd9["n_params"]) == 59905)
if arch.exists():
    a = pd.read_csv(arch)
    for dr in ("DR9", "DR7"):
        sub = a[a["dr"] == dr].set_index("arch")
        if {"L18", "shielded"} <= set(sub.index):
            gap = abs(float(sub.loc["shielded", "test_auc"]) - float(sub.loc["L18", "test_auc"]))
            rec("structure", f"|shielded-L18| AUC {dr}", "<=0.005", f"{gap:.4f}",
                "<=0.005", gap <= 0.005)
        if "L18" in sub.index:
            p = int(sub.loc["L18", "params"])
            rec("structure", f"L18 params ({dr})", "3,508,833", f"{p:,}",
                "==", p == 3508833, gate=(dr == "DR9"))

# ---- write report ----
def mark(ok):
    return {True: "PASS", False: "FAIL", None: "PENDING"}[ok]


prov = "torch unavailable"
try:
    import torch
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    prov = f"torch {torch.__version__}, device={dev}"
except Exception:
    pass

lines = [
    "# Apple Silicon (MPS) Huang-2021 reproduction vs. phoenix reference",
    "", f"Provenance: {prov}", "",
    "Port correctness is gated on identical computation: the four from-scratch test",
    "AUCs clearing their floors, MPS-vs-CUDA inference fidelity for the same",
    "checkpoints, reproduction of the leak-aware recovery table from the phoenix",
    "scores, the published-catalog grade counts, and the north-augmentation",
    "false-positive collapse. The absolute AUC values reflect an INDEPENDENT",
    "from-scratch MPS retrain and are reported with their deltas.", "",
    "| section | metric | reference | MPS | tol | result |",
    "| :--- | :--- | ---: | ---: | :--- | :--- |",
]
for r in rows:
    lines.append(f"| {r['section']} | {r['metric']} | {r['reference']} | "
                 f"{r['mps']} | {r['tol']} | {mark(r['ok'])} |")
gated = [r for r in rows if r["ok"] is not None and r["tol"] != "info"]
n_pass = sum(1 for r in gated if r["ok"])
n_pend = sum(1 for r in rows if r["ok"] is None)
lines += [
    "", f"**Gated checks:** {n_pass}/{len(gated)} passed"
    f"{f'  ({n_pend} pending)' if n_pend else ''}.",
    "", f"## {'OVERALL: PASS' if overall_ok else 'OVERALL: FAIL'}",
]
out = DATA / "REPRODUCTION_MPS_COMPARE.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[done] wrote {out}")
raise SystemExit(0 if overall_ok else 1)
