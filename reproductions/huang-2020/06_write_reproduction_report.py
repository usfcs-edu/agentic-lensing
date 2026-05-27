#!/usr/bin/env python3
"""
06_write_reproduction_report.py

Emit papers/REPRODUCTION.md collecting Phase-3a Huang-2020 reproduction
results: dataset, training-curve summary, best val AUC, test AUC, ROC.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "papers" / "REPRODUCTION.md"
OUT.parent.mkdir(exist_ok=True)


def load_json(p: Path) -> dict | list:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def fmt_int(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    hist = load_json(DATA / "training_history.json")
    test = load_json(DATA / "test_result.json")
    pos_count = len(pd.read_parquet(DATA / "positives_huang2020.parquet")) \
        if (DATA / "positives_huang2020.parquet").exists() else None
    neg_count = len(pd.read_parquet(DATA / "negatives.parquet")) \
        if (DATA / "negatives.parquet").exists() else None
    split_path = DATA / "training_split.parquet"
    split_counts = None
    if split_path.exists():
        sp = pd.read_parquet(split_path)
        split_counts = sp.groupby(["split", "label"]).size().unstack(fill_value=0)

    md: list[str] = []
    md.append("# Phase 3a Reproduction: Huang et al. 2020 (arXiv:1906.00970)")
    md.append("")
    md.append("**Paper:** \"Finding Strong Gravitational Lenses in the DESI DECam Legacy "
              "Survey\" (Huang, Storfer, Ravi, et al. 2020, ApJ 894 78). Uses the "
              "Lanusse 2018 *CMU DeepLens* ResNet-46 architecture (arXiv:1703.02642) "
              "re-implemented in TensorFlow; we re-implemented the same in PyTorch "
              "(script 01) following Lanusse §3.1 Fig 4.")
    md.append("")
    md.append("**Reproduction scope:** Phase 3a = train the network from paper + public "
              "code only (NO Huang training code). Uses the NeuraLens-released candidate "
              "catalog as positives + DESI DR1 zcat-sampled galaxies as negatives. The "
              "full DECaLS deployment (sweep ~10⁶ galaxies → rank by ResNet score → human "
              "VI of top scores) is Phase 3b and explicitly out of scope here.")
    md.append("")

    md.append("## 1. Training set")
    md.append("")
    md.append("| Source | Count |")
    md.append("|---|---:|")
    if pos_count is not None:
        md.append(f"| NeuraLens L18 lens candidates (positives) | {fmt_int(pos_count)} |")
    if neg_count is not None:
        md.append(f"| DR1 zcat galaxies in DECaLS footprint (negatives) | {fmt_int(neg_count)} |")
    if split_counts is not None:
        md.append("")
        md.append("Split (70:20:10 train/val/test, stratified by label):")
        md.append("")
        md.append("```")
        md.append(str(split_counts))
        md.append("```")
    md.append("")
    md.append("Cutout spec: 101×101 px grz at 0.262″/px ≈ 26.5″ FoV, `ls-dr9` layer, "
              "from `legacysurvey.org/viewer/fits-cutout`. Same size as Huang+2020 §3.2.")
    md.append("")
    md.append("**Caveats versus Huang+2020's *actual* training set:**")
    md.append("- They used 613 known lenses from Master Lens Database + 6 specific recent "
              "publications (Carrasco, Diehl, Pourrahmani, Sonnenfeld, Wong, Jacobs); we "
              "use the NeuraLens-published L18 candidate list (949 systems) instead, "
              "which is their *output* and therefore includes some of their *input* "
              "training examples plus the newly discovered candidates.")
    md.append("- They used 13,000 non-lens cutouts in DECaLS with curated by-eye hard "
              "negatives (spirals, galaxy groups, cosmic rays, artifacts); we use 5,000 "
              "uncurated DR1 galaxies. Higher false-positive rate during deployment is "
              "expected.")
    md.append("- 5.3:1 class imbalance vs Huang+2020's 21:1.")
    md.append("")

    md.append("## 2. Architecture")
    md.append("")
    md.append("Lanusse 2018 / CMU DeepLens ResNet-46 (1 stem conv + 5 stages × 3 "
              "pre-activated bottleneck blocks). 3.5 M parameters at 3-channel input.")
    md.append("")
    md.append("Per-stage spatial resolution for 101×101 input: 101 → 51 → 26 → 13 → 7 → "
              "AvgPool(1) → FC(512→1) → Sigmoid.")
    md.append("")

    md.append("## 3. Hyperparameters (Lanusse §3.4 + Huang §3.2)")
    md.append("")
    md.append("- Optimizer: **Adam**, default β")
    md.append("- Initial LR: **1e-3**, divided by 10 every 40 epochs")
    md.append("- Batch size: **128**")
    md.append("- Epochs: **120**")
    md.append("- Loss: **BCE with logits** (numerically stable)")
    md.append("- Preprocessing: per-band mean subtraction + std normalisation, clip ±250σ")
    md.append("- Augmentation: random ±90° rotation + flip + zoom [0.9, 1.0]")
    md.append("")

    md.append("## 4. Results")
    md.append("")
    if hist:
        best = max(hist, key=lambda r: r.get("val_auc", -1) if r.get("val_auc") is not None else -1)
        md.append(f"- Best validation AUC: **{best['val_auc']:.4f}** (at epoch "
                  f"{best['epoch']})")
        md.append(f"- Final epoch train loss: {hist[-1]['train_loss']:.4f}")
        md.append(f"- Final epoch val loss: {hist[-1]['val_loss']:.4f}")
        md.append(f"- Total training wall-clock: {hist[-1]['elapsed_s']/60:.1f} min")
    if test:
        md.append(f"- **Held-out test AUC: {test['test_auc']:.4f}** "
                  f"(n_test={test['n_test']:,}, checkpoint epoch {test['best_epoch']})")
        md.append("")
        md.append("Compare to Huang+2020 published validation AUC = 0.98 (their 70:20:10 "
                  "split on the 613-positive / 13,000-negative set on NERSC Cori).")
    if not hist and not test:
        md.append("*Training history not yet generated. Run "
                  "`05_train_resnet.py` then re-run this script.*")
    md.append("")

    md.append("## 5. Reproducibility")
    md.append("")
    md.append("- NeuraLens catalog: `https://drive.google.com/file/d/1_KbEHWhl8LeeTyXpXkWFbLRxt6o42wBg/view`")
    md.append("- DR1 zcat (for negatives): `https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits`")
    md.append("- Cutout service: `https://www.legacysurvey.org/viewer/fits-cutout?layer=ls-dr9&size=101&pixscale=0.262&bands=grz`")
    md.append("- Lanusse 2018 reference: `https://github.com/McWilliamsCenter/CMUDeepLens` (Theano/Lasagne; this PyTorch port is in `01_lanusse_resnet.py`)")
    md.append("- venv: `/raid/benson/.venvs/lensfinder` (torch 2.12 + torchvision + astropy)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("*Generated by `06_write_reproduction_report.py`. "
              "Source scripts in `reproductions/huang-2020/*.py`, run logs in `*.log`, "
              "JSON artifacts in `data/{training_history,test_result}.json`.*")
    md.append("")

    OUT.write_text("\n".join(md))
    print(f"[done] wrote {OUT}  ({len(md)} lines)")


if __name__ == "__main__":
    main()
