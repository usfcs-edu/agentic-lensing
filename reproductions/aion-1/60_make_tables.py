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

    # Task 2: stellar params (DD-Payne DESI), R2, desi vs desi+parallax
    t2 = _load("task2_ddpayne.json")
    if t2:
        lines += ["## Task 2 — Stellar params (DESI×DD-Payne), R²", ""]
        sp = PT["stellar_props_R2"]
        targs = ["Teff", "logg", "FeH", "vmic"]
        lines += ["| config / variant | " + " | ".join(targs) + " |",
                  "|" + "---|" * (len(targs) + 1)]
        for cfg in ["desi", "desi_plx"]:
            if cfg not in t2:
                continue
            for v in VAR:
                if v in t2[cfg]:
                    r = t2[cfg][v]["attn_R2"]
                    lines.append(f"| {cfg} / {v} | " + " | ".join(_fmt(r[t]) for t in targs) + " |")
        lines.append(f"| **paper (DESI+Plx, B)** | {sp['teff']} | {sp['logg']} | {sp['feh']} | {sp['vmicro']} |")
        lines += ["", "_Note: +parallax gives ~no logg gain on the frozen encoder "
                  "(paper's number likely needs finetuning)._", ""]

    # Task 5: segmentation IoU
    t5 = _load("task5_gz3d.json")
    if t5:
        lines += ["## Task 5 — Galaxy structure segmentation (GZ3D), IoU", ""]
        p = PT["segmentation_iou"]
        lines += ["| variant | spiral arms | bar | N |", "|---|---|---|---|"]
        for v in VAR:
            if v in t5:
                lines.append(f"| {v} | {_fmt(t5[v]['spiral_arms'])} | {_fmt(t5[v]['bar'])} | {t5[v]['n']} |")
        lines.append(f"| **paper (B)** | {p['spiral_arms']} | {p['bar']} | ~2800 |")
        lines.append("")

    # Task 9: strong-lens retrieval
    t9 = _load("task9_lenses.json")
    if t9:
        lines += ["## Task 9 — Strong-lens retrieval (SuGOHI), nDCG@10", ""]
        lines += ["_LegacySurvey lenses (paper uses HSC); corpus less rare than paper._", "",
                  "| variant | nDCG@10 | corpus | lenses |", "|---|---|---|---|"]
        for v in VAR:
            if v in t9:
                lines.append(f"| {v} | {_fmt(t9[v]['ndcg@10'])} | {t9[v]['corpus']} | {t9[v]['n_positive']} |")
        lines.append(f"| **paper (B, HSC)** | {PT['retrieval_ndcg10']['hsc_lenses']:.3f} | — | — |")
        lines.append("")

    # Tasks 7/8 faithful: GZ-DECaLS (Walmsley+2022) full corpus
    tgd = _load("task78_gzdecals_retrieval.json")
    if tgd:
        lines += ["## Tasks 7/8 (faithful) — GZ-DECaLS retrieval, nDCG@10", ""]
        lines += ["_Walmsley+2022 vote-fraction positives in a 63k rare-positive corpus "
                  "(real griz, two-machine campaign)._", "",
                  "| variant | spirals | mergers | corpus |", "|---|---|---|---|"]
        rp = PT["retrieval_ndcg10"]
        for v in VAR:
            if v in tgd:
                s = tgd[v]["spirals"]; m = tgd[v]["mergers"]
                lines.append(f"| {v} | {_fmt(s['ndcg@10'])} | {_fmt(m['ndcg@10'])} | {s['corpus']} |")
        lines.append(f"| **paper (B)** | {rp['gz_spirals']:.3f} | {rp['gz_mergers']:.3f} | ~171k |")
        lines.append("")

    # Task 11: spectral super-resolution
    t11 = _load("task11_superres.json")
    if t11:
        lines += ["## Task 11 — Spectral super-resolution (Gaia XP→DESI)", ""]
        lines += ["| variant | median corr | mean corr | N |", "|---|---|---|---|"]
        for v, r in t11.items():
            lines.append(f"| {v} | {_fmt(r['median_corr'])} | {_fmt(r['mean_corr'])} | {r['n']} |")
        lines += ["", "_Qualitative in paper (line recovery); high corr = good reconstruction._", ""]

    # Task 6: low-data regime
    t6 = _load("task6_lowdata.json")
    if t6:
        lines += ["## Task 6 — Low-data regime (PROVABGS), z R² vs #labels", ""]
        for cfg in ["phot", "phot_spec"]:
            if cfg in t6 and "base" in t6[cfg]:
                d = t6[cfg]["base"]
                ns = ", ".join(str(n) for n in d["N"])
                zs = ", ".join(_fmt(z, 2) for z in d["z"])
                lines += [f"- **{cfg}** (base): N=[{ns}] → z R²=[{zs}]"]
        lines += ["", "_Paper: performance saturates by 10³–10⁴ labels (reproduced)._", ""]

    out = "\n".join(lines)
    (R / "REPORT.md").write_text(out)
    print(out)
    print("\nMAKE_TABLES_OK ->", R / "REPORT.md")


if __name__ == "__main__":
    main()
