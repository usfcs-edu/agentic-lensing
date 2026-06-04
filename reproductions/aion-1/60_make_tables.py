"""
60 -- Assemble the headline comparison tables (ours vs paper) across B/L/XL.

Reads every data/results/task*.json that exists and renders a Markdown report to
data/results/REPORT.md (and prints it). Missing tasks are shown as TODO so the
report is always buildable mid-run. Paper targets come from _config.paper_targets().

Run: python 60_make_tables.py
"""

import json

import _config as C

R = C.RESULTS
PT = C.paper_targets()
VAR = C.VARIANTS


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    lines = ["# AION-1 reproduction — headline results", ""]
    lines += ["Frozen released checkpoints (`polymathic-ai/aion-{base,large,xlarge}`), "
              "our probes vs the paper's printed AION numbers. SEED=2026.", ""]

    # Task 1: galaxy properties (R2), per config per variant
    t1 = _load("task1_provabgs.json")
    if t1:
        lines += ["## Task 1 — Galaxy property estimation (PROVABGS), R²", ""]
        targs = ["z", "logmass", "age", "logZ", "sSFR"]
        paper = PT["galaxy_props_R2"]
        lines += ["| config / variant | " + " | ".join(targs) + " |",
                  "|" + "---|" * (len(targs) + 1)]
        for cfg in ["phot", "phot_image", "phot_spec", "phot_image_spec"]:
            if cfg not in t1:
                continue
            for v in VAR:
                if v not in t1[cfg]:
                    continue
                row = t1[cfg][v]["attn_R2"]
                lines.append(f"| {cfg} / {v} | " +
                             " | ".join(_fmt(row[t]) for t in targs) + " |")
        lines.append("| **paper (phot+im+spec, B)** | " +
                     " | ".join(_fmt(paper[t], 2) for t in targs) + " |")
        lines.append("")

    # Task 3: APOGEE residual std
    t3 = _load("task3_gaia_apogee.json")
    if t3:
        lines += ["## Task 3 — APOGEE×GaiaXP stellar params (residual std)", ""]
        p = PT["apogee_resid_std"]
        lines += ["| variant | Teff (K) | logg (dex) | [Fe/H] (dex) | N |",
                  "|---|---|---|---|---|"]
        for v in VAR:
            if v in t3:
                r = t3[v]["attn_residual_std"]
                lines.append(f"| {v} | {_fmt(r['Teff'],1)} | {_fmt(r['logg'])} | "
                             f"{_fmt(r['FeH'])} | {t3[v]['n_train']+t3[v]['n_test']} |")
        lines.append(f"| **paper (B)** | {p['teff_K']} | {p['logg_dex']} | {p['feh_dex']} | ~10000 |")
        lines.append("")

    # Task 4: morphology accuracy
    t4 = _load("task4_gz10.json")
    if t4:
        lines += ["## Task 4 — Galaxy morphology (Galaxy10 DECaLS), accuracy", ""]
        lines += ["| variant | accuracy | N |", "|---|---|---|"]
        for v in VAR:
            if v in t4:
                lines.append(f"| {v} | {_fmt(t4[v]['accuracy'])} | "
                             f"{t4[v]['n_train']+t4[v]['n_test']} |")
        lines.append(f"| **paper (B)** | {PT['morphology_acc']:.3f} | ~8000 |")
        lines.append("")

    # Tasks 7/8: retrieval
    t78 = _load("task78_gz10_retrieval.json")
    if t78:
        lines += ["## Tasks 7/8 — Morphology retrieval (Galaxy10 DECaLS), nDCG@10", ""]
        lines += ["_Corpus differs from paper's full GZ-DECaLS; best-effort._", "",
                  "| variant | spirals | mergers |", "|---|---|---|"]
        for v in VAR:
            if v in t78:
                sp = t78[v].get("spirals", {}).get("ndcg@10")
                mg = t78[v].get("mergers", {}).get("ndcg@10")
                lines.append(f"| {v} | {_fmt(sp)} | {_fmt(mg)} |")
        rp = PT["retrieval_ndcg10"]
        lines.append(f"| **paper (B)** | {rp['gz_spirals']:.3f} | {rp['gz_mergers']:.3f} |")
        lines.append("")

    # Task 10: redshift posterior
    t10 = _load("task10_redshift_posterior.json")
    if t10:
        lines += ["## Task 10 — Redshift posterior (generative)", ""]
        lines += ["| variant/config | point R² | mean post. std | NLL@true |",
                  "|---|---|---|---|"]
        for v, cfgs in t10.items():
            for cfg, r in cfgs.items():
                lines.append(f"| {v}/{cfg} | {_fmt(r['point_R2'])} | "
                             f"{_fmt(r['mean_post_std'],4)} | {_fmt(r['nll_true'])} |")
        lines.append("")

    out = "\n".join(lines)
    (R / "REPORT.md").write_text(out)
    print(out)
    print("\nMAKE_TABLES_OK ->", R / "REPORT.md")


if __name__ == "__main__":
    main()
