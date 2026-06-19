#!/usr/bin/env python3
"""Build data/dr11s_desi_resolution_AB.csv: the DESI-resolution-only grade-A/B candidates from a
cascade run -- graded A/B at DESI tier-1 with NO HSC/Euclid high-resolution coverage (so unvalidatable
beyond DECaLS 1"). Default source is the honest-residual run dr11s_cascade_full_resid2.parquet.

"No high-res coverage" = the candidate never reached HSC tier-2 in ANY of the three cascade runs
(prior / legacy-now / new); a candidate that is HSC-covered (reached tier-2 in some run) is excluded
even if it did not escalate in this particular run. This reproduces the original 78-list filter
(A/B & not-HSC-covered) and excludes the one covered edge case the legacy list dropped.

  PYTHONPATH=<reproductions> /home2/benson/.venvs/lensjudge/bin/python dr11-campaign/build_resolution_list.py \
      [--parquet lensjudge/outputs/dr11s_cascade_full_resid2.parquet]
"""
import argparse
from pathlib import Path
import pandas as pd

REPRO = Path("/home2/benson/git/agentic-lensing/reproductions")
LJ = REPRO / "lensjudge"
CAMP = REPRO / "dr11-campaign"
AB = {"A", "B"}


def load(p):
    return pd.read_parquet(p).assign(name=lambda d: d["name"].astype(str)).set_index("name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(LJ / "outputs/dr11s_cascade_full_resid2.parquet"))
    ap.add_argument("--out", default=str(CAMP / "data/dr11s_desi_resolution_AB.csv"))
    a = ap.parse_args()

    new = load(a.parquet)
    # HSC-covered = reached tier-2 in ANY run (a sky-position property, not run-specific)
    covered = set()
    for f in ("dr11s_cascade_full.parquet", "dr11s_cascade_full_legacy_now.parquet",
              "dr11s_cascade_full_resid2.parquet"):
        d = load(LJ / "outputs" / f)
        covered |= set(d.index[d["highres_survey"].notna()])

    sel = new[(new["grade_pred"].isin(AB)) & (new["highres_survey"].isna())].copy()
    sel = sel[~sel.index.isin(covered)]                     # drop HSC-covered-but-unescalated
    sel = sel.sort_values("p_lens", ascending=False)        # rank by DESI tier-1 p_lens

    man = pd.read_csv(LJ / "manifests_dr11s_cand500.csv").astype({"name": str}).set_index("name")
    rows = []
    for i, (nm, r) in enumerate(sel.iterrows(), 1):
        m = man.loc[nm]
        sc = r.get("stage1_contaminant")
        rows.append(dict(rank=i, name=nm, ra=float(m["ra"]), dec=float(m["dec"]),
                         grade_pred=r["grade_pred"], p_lens=round(float(r["p_lens"]), 2),
                         p_meta=float(m["p_meta"]),
                         stage1_contaminant=("" if pd.isna(sc) else sc)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    nA = int((out.grade_pred == "A").sum()); nB = int((out.grade_pred == "B").sum())
    print(f"wrote {a.out}: {len(out)} DESI-resolution-only A/B  (A={nA} B={nB})")

    # composition vs the previous (legacy) 78-list, for the prose update
    prev_p = CAMP / "data" / "dr11s_desi_resolution_AB.PREV.csv"
    if prev_p.exists():
        prev = set(pd.read_csv(prev_p).astype({"name": str}).name)
        cur = set(out.name)
        print(f"  vs prev list ({len(prev)}): kept={len(cur & prev)} added={len(cur - prev)} dropped={len(prev - cur)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
