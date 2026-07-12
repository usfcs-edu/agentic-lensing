#!/usr/bin/env python3
"""C3 SELECTION slice: a stratified valsel sample for the matched-vs-direct A/B.

Selection artifact — this slice exists to (a) measure whether the matched
information set beats plain direct on the same rows and (b) fit the matched
operating point (calibrate.py). It draws ONLY from parity_train_pool split=valsel;
the frozen gate stays untouched for the one-shot final evaluation.

  python lensjudge/parity/build_c3_valsel_slice.py

Output: outputs/c3_valsel_slice.csv (LensBench manifest format, SHA-pinned).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
OUT = Path(__file__).resolve().parents[1] / "outputs"
N = {"A": 30, "B": 30, "C": 30, "D": 40, "random": 20}


def main() -> None:
    pool = pd.read_csv(OUT / "parity_train_pool.csv")
    val = pool[pool.split == "valsel"]
    rng = np.random.default_rng(SEED)
    parts = []
    for grade in ("A", "B", "C"):
        g = val[(val.label_source == "graded") & (val.grade == grade)]
        parts.append(g.sample(min(N[grade], len(g)), random_state=SEED))
    d = val[val.label_source == "graded_D"]
    parts.append(d.sample(min(N["D"], len(d)), random_state=SEED))
    r = val[val.label_source == "random_neg"]
    parts.append(r.sample(min(N["random"], len(r)), random_state=SEED))
    s = pd.concat(parts, ignore_index=True)

    man = pd.DataFrame({
        "name": s.name, "ra": s.ra, "dec": s.dec, "survey_key": s.survey_key,
        "grade_truth": s.grade.where(s.grade.astype(str).str.len() > 0, "D"),
        "binary_label": np.where(s.label_source.eq("graded") , "lens", "nonlens"),
        "source": s.label_source.map({"graded": "graded", "graded_D": "graded_D",
                                      "random_neg": "random_neg"}),
        "region": "?", "tractor_type": "?", "p_meta": s.p_pub, "leak": "no",
    })
    out = OUT / "c3_valsel_slice.csv"
    man.to_csv(out, index=False)
    h = hashlib.sha256(man.to_csv(index=False).encode()).hexdigest()[:16]
    Path(str(out) + ".sha").write_text(h + "\n")
    print(f"[slice] {len(man)} rows -> {out} (sha {h})")
    print("  by source:", man.source.value_counts().to_dict())
    print("  by grade:", man.grade_truth.value_counts().to_dict())


if __name__ == "__main__":
    main()
