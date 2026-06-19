#!/usr/bin/env python3
"""Paired A/B comparison of the LensJudge residual rewrite on the DR11-south 500.

Three cascade runs over the SAME manifests_dr11s_cand500.csv:
  PRIOR  outputs/dr11s_cascade_full.parquet          legacy residual, Jun-17 (stored baseline)
  LEG    outputs/dr11s_cascade_full_legacy_now.parquet  legacy residual, re-run now (nondeterminism control)
  NEW    outputs/dr11s_cascade_full_resid2.parquet      new signed-chi residual, re-run now

Questions:
  1. Nondeterminism floor: PRIOR vs LEG (same legacy residual, different run) -- how much do
     grades/A-B selections move from LLM nondeterminism alone?
  2. Residual effect: LEG vs NEW (paired, same code/period) -- escalation-set change, grade
     confusion, A/B set diff, p_lens shift. This is the clean attribution of the residual rewrite.
  3. Catalog integrity: do the 24 confirmed A/B (esp. the 9 NEW) survive under the honest residual?
  4. New promotions/demotions: C/D<->A/B flips that reach HSC tier-2.
  5. Matched-FPR diagnostic: apply the per-arm calibrated operating points (legacy thr=0.125,
     new thr=0.20) to the tier-1 grade p_lens, so the hotter new grader is compared at equal FPR.

Run (lensjudge venv):
  PYTHONPATH=<reproductions> /home2/benson/.venvs/lensjudge/bin/python dr11-campaign/compare_residual_ab.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

LJ = Path("/home2/benson/git/agentic-lensing/reproductions/lensjudge")
CAMP = Path("/home2/benson/git/agentic-lensing/reproductions/dr11-campaign")
OUT = CAMP / "data"

ARMS = {
    "PRIOR": LJ / "outputs" / "dr11s_cascade_full.parquet",          # legacy, Jun-17 stored
    "LEG":   LJ / "outputs" / "dr11s_cascade_full_legacy_now.parquet",  # legacy, now
    "NEW":   LJ / "outputs" / "dr11s_cascade_full_resid2.parquet",      # new, now
}
OP = {
    "legacy": LJ / "eval" / "residual_op_point_legacy.json",
    "new":    LJ / "eval" / "residual_op_point_new.json",
}
AB = {"A", "B"}


def load_arm(p: Path):
    if not p.exists():
        return None
    df = pd.read_parquet(p).copy()
    df["name"] = df["name"].astype(str)
    return df.set_index("name")


def summary(name, df):
    if df is None:
        return f"  {name:6s}  (not present yet: {ARMS[name].name})"
    esc = int(df.get("escalated", pd.Series(False, index=df.index)).fillna(False).sum())
    t2mask = df["highres_survey"].notna() if "highres_survey" in df else pd.Series(False, index=df.index)
    t2 = int(t2mask.sum())
    gd = df["grade_pred"].value_counts().reindex(["A", "B", "C", "D"]).fillna(0).astype(int)
    nab = int(df["grade_pred"].isin(AB).sum())                       # raw grade A/B (incl. tier-1; shows hotness)
    t2ab = int((t2mask & df["grade_pred"].isin(AB)).sum())            # tier-2 HSC-confirmed A/B (the catalog metric)
    cost = float(df.get("cost_usd", pd.Series(dtype=float)).sum())
    return (f"  {name:6s}  n={len(df):3d}  esc={esc:3d}  tier2(hsc)={t2:3d}  "
            f"A={gd['A']:3d} B={gd['B']:3d} C={gd['C']:3d} D={gd['D']:3d}  "
            f"rawA/B={nab:3d}  tier2-A/B={t2ab:3d}  ${cost:6.2f}")


def grade_confusion(a, b, la, lb):
    """Confusion of grade_pred: rows=la's grade, cols=lb's grade, over common names."""
    common = a.index.intersection(b.index)
    ca = a.loc[common, "grade_pred"]
    cb = b.loc[common, "grade_pred"]
    ct = pd.crosstab(ca, cb).reindex(index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"]).fillna(0).astype(int)
    return ct, common


def set_jaccard(sa, sb):
    sa, sb = set(sa), set(sb)
    u = sa | sb
    return (len(sa & sb) / len(u)) if u else 1.0, sa, sb


def ab_set(df):
    return set(df.index[df["grade_pred"].isin(AB)])


def esc_set(df):
    if "escalated" not in df:
        return set()
    return set(df.index[df["escalated"].fillna(False)])


def t2_set(df):
    if "highres_survey" not in df:
        return set()
    return set(df.index[df["highres_survey"].notna()])


def matched_fpr_pass(df, op_path):
    """Apply the calibrated op-point threshold to the tier-1 grade p_lens (stage1_p_lens),
    the score the op-points were fit on (the direct grader). Returns the passing name set."""
    thr = json.load(open(op_path))["threshold"]
    col = "stage1_p_lens" if "stage1_p_lens" in df else "p_lens_tier1"
    s = pd.to_numeric(df[col], errors="coerce").fillna(-1)
    return set(df.index[s >= thr]), thr, col


def main():
    dfs = {k: load_arm(p) for k, p in ARMS.items()}

    print("=" * 96)
    print("DR11-south 500 -- LensJudge residual A/B  (PRIOR=legacy/Jun17, LEG=legacy/now, NEW=new/now)")
    print("=" * 96)
    print("\n[1] Per-arm summary")
    for k in ("PRIOR", "LEG", "NEW"):
        print(summary(k, dfs[k]))

    have_leg, have_new = dfs["LEG"] is not None, dfs["NEW"] is not None

    # ---- (1) nondeterminism floor: PRIOR vs LEG ----
    if have_leg:
        print("\n[2] Nondeterminism floor  (PRIOR vs LEG -- same legacy residual, different run)")
        ct, common = grade_confusion(dfs["PRIOR"], dfs["LEG"], "PRIOR", "LEG")
        agree = int(np.trace(ct.values)) / int(ct.values.sum())
        print(f"    grade agreement on {len(common)} common: {agree:.1%}")
        jac, sa, sb = set_jaccard(ab_set(dfs["PRIOR"]), ab_set(dfs["LEG"]))
        print(f"    A/B set: PRIOR={len(sa)} LEG={len(sb)} | shared={len(sa & sb)} Jaccard={jac:.2f}")
        print(f"    A/B only in PRIOR (lost on rerun): {sorted(sa - sb)}")
        print(f"    A/B only in LEG   (gained on rerun): {sorted(sb - sa)}")

    # ---- (2) residual effect: LEG vs NEW ----
    if have_leg and have_new:
        print("\n[3] Residual effect  (LEG vs NEW -- paired, isolates the residual rewrite)")
        # escalation
        jce, eL, eN = set_jaccard(esc_set(dfs["LEG"]), esc_set(dfs["NEW"]))
        print(f"    escalation set: LEG={len(eL)} NEW={len(eN)} shared={len(eL & eN)} Jaccard={jce:.2f}")
        # tier-2
        jct, tL, tN = set_jaccard(t2_set(dfs["LEG"]), t2_set(dfs["NEW"]))
        print(f"    tier-2(hsc) set: LEG={len(tL)} NEW={len(tN)} shared={len(tL & tN)} Jaccard={jct:.2f}")
        print(f"      tier-2 only in NEW: {sorted(tN - tL)}")
        print(f"      tier-2 only in LEG: {sorted(tL - tN)}")
        # grade confusion
        ct, common = grade_confusion(dfs["LEG"], dfs["NEW"], "LEG", "NEW")
        print("    grade_pred confusion (rows=LEG, cols=NEW):")
        print(ct.to_string().replace("\n", "\n      "))
        # A/B set diff
        jab, abL, abN = set_jaccard(ab_set(dfs["LEG"]), ab_set(dfs["NEW"]))
        print(f"    A/B set: LEG={len(abL)} NEW={len(abN)} shared={len(abL & abN)} Jaccard={jab:.2f}")
        print(f"      A/B only in NEW (honest-residual promotions): {sorted(abN - abL)}")
        print(f"      A/B only in LEG (honest-residual demotions):  {sorted(abL - abN)}")
        # p_lens shift on common escalated
        com = sorted(eL & eN)
        if com:
            dl = pd.to_numeric(dfs["NEW"].loc[com, "p_lens"], errors="coerce") - \
                 pd.to_numeric(dfs["LEG"].loc[com, "p_lens"], errors="coerce")
            print(f"    p_lens shift (NEW-LEG) on {len(com)} common escalated: "
                  f"mean={dl.mean():+.3f} median={dl.median():+.3f} (hotter=>positive)")

    # ---- (3) confirmed catalog integrity ----
    confp = OUT / "dr11s_confirmed_AB.csv"
    if confp.exists():
        conf = pd.read_csv(confp).astype({"name": str})
        clscol = "class" if "class" in conf else None
        print(f"\n[4] Confirmed A/B catalog integrity ({len(conf)} systems)")
        rows = []
        for _, r in conf.iterrows():
            nm = r["name"]; rec = {"name": nm, "class": (r[clscol] if clscol else "?")}
            for k in ("PRIOR", "LEG", "NEW"):
                d = dfs[k]
                if d is not None and nm in d.index:
                    rec[f"{k}_grade"] = d.loc[nm, "grade_pred"]
                    rec[f"{k}_plens"] = round(float(d.loc[nm, "p_lens"]), 2)
                else:
                    rec[f"{k}_grade"] = "-"; rec[f"{k}_plens"] = np.nan
            rec["survives_NEW"] = rec.get("NEW_grade") in AB
            rows.append(rec)
        tab = pd.DataFrame(rows)
        print(tab.to_string(index=False))
        if have_new:
            dropped = tab[(~tab["survives_NEW"]) & (tab["NEW_grade"] != "-")]
            print(f"\n    confirmed A/B that FALL BELOW A/B under NEW: {len(dropped)}")
            if len(dropped):
                print(dropped[["name", "class", "PRIOR_grade", "NEW_grade", "NEW_plens"]].to_string(index=False))
            if clscol:
                for cls in sorted(tab["class"].unique()):
                    sub = tab[tab["class"] == cls]
                    surv = int(sub["survives_NEW"].sum())
                    print(f"    class={cls}: {surv}/{len(sub)} survive as A/B under NEW")
        tab.to_csv(OUT / "residual_ab_confirmed_tracking.csv", index=False)
        print(f"    -> wrote {OUT/'residual_ab_confirmed_tracking.csv'}")

    # ---- (4) new promotions/demotions reaching tier-2 ----
    if have_new:
        new, prior = dfs["NEW"], dfs["PRIOR"]
        common = new.index.intersection(prior.index)
        promo = [n for n in common if new.loc[n, "grade_pred"] in AB
                 and prior.loc[n, "grade_pred"] not in AB
                 and pd.notna(new.loc[n].get("highres_survey"))]
        demo = [n for n in common if new.loc[n, "grade_pred"] not in AB
                and prior.loc[n, "grade_pred"] in AB]
        print(f"\n[5] vs PRIOR: {len(promo)} new A/B promotions reaching HSC tier-2; {len(demo)} demotions")
        for n in sorted(promo):
            print(f"    PROMO {n}: PRIOR={prior.loc[n,'grade_pred']} -> NEW={new.loc[n,'grade_pred']} "
                  f"p_lens={new.loc[n,'p_lens']:.2f} ({new.loc[n,'highres_survey']})")
        for n in sorted(demo):
            print(f"    DEMOTE {n}: PRIOR={prior.loc[n,'grade_pred']} -> NEW={new.loc[n,'grade_pred']}")

    # ---- (5) matched-FPR diagnostic on tier-1 grade ----
    if have_leg and have_new:
        print("\n[6] Matched-FPR diagnostic (per-arm op-point on the tier-1 grade p_lens)")
        pL, thrL, colL = matched_fpr_pass(dfs["LEG"], OP["legacy"])
        pN, thrN, colN = matched_fpr_pass(dfs["NEW"], OP["new"])
        print(f"    LEG  pass tier-1 @ thr={thrL} ({colL}): {len(pL)}")
        print(f"    NEW  pass tier-1 @ thr={thrN} ({colN}): {len(pN)}")
        print(f"    shared={len(pL & pN)}  only-NEW={len(pN - pL)}  only-LEG={len(pL - pN)}")
        print("    (op-points fit on the benchmark direct-grader p_lens; applied to tier-1 grade only,")
        print("     NOT the tier-2 HSC p_lens that drives the final catalog -- diagnostic, not the catalog.)")

    print("\n" + "=" * 96)
    print("done" + ("" if (have_leg and have_new) else "  [PARTIAL -- LEG/NEW arms not finished yet]"))


if __name__ == "__main__":
    main()
