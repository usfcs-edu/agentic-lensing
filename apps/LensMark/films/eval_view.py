"""Column view of a campaign's eval table (exports/eval/runs.csv): the ten headline columns.
A disclosed selection - the full pandas table has 19 columns and wraps on camera."""
import sys

import pandas as pd

df = pd.read_csv(f"{sys.argv[1]}/exports/eval/runs.csv")
cols = ["model", "effort", "proposed", "TP", "FP", "FN",
        "precision_strict", "recall", "spurious_mask_rate", "mean_cost_usd"]
out = df[[c for c in cols if c in df.columns]].copy()
for c in out.select_dtypes("float").columns:
    out[c] = out[c].round(3)
print(out.to_string(index=False, na_rep="—"))
