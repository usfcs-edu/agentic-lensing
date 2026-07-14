#!/usr/bin/env python3
"""finetune/augment_sft_v5.py — run-3 corpus: dihedral augmentation + random-field dilution.

Takes the v5 SFT train jsonl and produces sft_v5r3_train.jsonl with:
  1. every original row;
  2. ONE deterministic dihedral copy per row (hash-picked from the 7 non-identity transforms,
     applied to all 4 rendered views; same target — the label is orientation-invariant, which is
     exactly the inductive bias we want taught);
  3. the ~1,500 random-field negatives NOT used by bench v5.1 (pool-covariate dilution from the
     Phase-E lesson), rendered from the hsc_v5 cache with rf-specific per-example-unique targets
     (the _v5_target nonlens stems claim "committee rejected", which would be a false fact here).

Val set is unchanged (sft_v5_val.jsonl); selection stays on valsel-180 as before.

  LENSJUDGE_HSC_CACHE=lensjudge/cache/hsc_v5 python -m lensjudge.finetune.augment_sft_v5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.finetune.distill_hsc import (  # noqa: E402
    OUT, SEED, DIRECT_SYS, _hash01, _hsc_imgs, _target_json,
)
from lensjudge.common import render  # noqa: E402

DIHEDRAL = [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
            Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]
AUG_DIR = OUT / "images_v5r3"
USER_MSG = ("<image><image><image><image>\nGrade this strong-lens candidate at HSC 0.168\"/px. "
            "Rendered grizy views (full, lum, zoom, lum_sub). Respond with ONLY the JSON.")


def _rf_target(name: str, ra: float, dec: float) -> str:
    """Random-field analogue of _v5_target: unique, fact-bearing, and truthful (no committee claim)."""
    h = lambda s: _hash01(name, s)  # noqa: E731
    p = round(0.01 + h("p") * 0.03, 2)
    crit = {k: int(h(f"c{i}") * 2) for i, k in enumerate(
        ("blue_source", "low_surface_brightness", "curvature", "counter_images", "arc_morphology"))}
    conf = round(0.75 + h("cf") * 0.2, 2)
    stems = [f"Ordinary field galaxy at ({ra:.4f},{dec:.4f}); no arc-like or lensed features.",
             f"({ra:.4f},{dec:.4f}): unremarkable field scene, nothing consistent with lensing.",
             f"Field position ({ra:.4f},{dec:.4f}) shows no deflector-arc geometry."]
    return _target_json("D", p_lens=p, rationale=stems[int(h('r') * len(stems))], crit=crit,
                        confidence=conf, contaminant=None)


def main():
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(x) for x in open(OUT / "sft_v5_train.jsonl")]
    out = list(recs)

    # 1+2: dihedral copy per original row
    for n, rec in enumerate(recs):
        stem0 = Path(rec["images"][0]).stem
        t = int(_hash01(stem0, "dihedral") * len(DIHEDRAL))
        new_paths = []
        for ip in rec["images"]:
            ip = Path(ip)
            op = AUG_DIR / f"{ip.stem}_d{t}.png"
            if not op.exists():
                Image.open(ip).transpose(DIHEDRAL[t]).save(op)
            new_paths.append(str(op))
        out.append({**rec, "images": new_paths})
        if (n + 1) % 1000 == 0:
            print(f"  dihedral {n + 1}/{len(recs)}")

    # 3: random-field dilution (exclude the 500 bench rows)
    rf = pd.read_csv(config.HERE / "cache" / "v5_catalogs" / "hsc_v51_randomfield.csv")
    bench = pd.read_csv(config.OUT / "bench_v51" / "bench_v51.csv")
    rf = rf[~rf.name.isin(set(bench[bench.pool == "random_field"].name))]
    print(f"[rf] {len(rf)} random-field rows (bench excluded)")
    miss = 0
    for r in rf.itertuples():
        imgs = _hsc_imgs(r.ra, r.dec)
        if not imgs:
            miss += 1
            continue
        safe = str(r.name).replace("/", "_")
        paths = []
        for v, img in imgs.items():
            p = AUG_DIR / f"{safe}_{v}.png"
            if not p.exists():
                render.save_png(img, p)
            paths.append(str(p))
        out.append({"messages": [{"role": "system", "content": DIRECT_SYS},
                                 {"role": "user", "content": USER_MSG},
                                 {"role": "assistant", "content": _rf_target(str(r.name), r.ra, r.dec)}],
                    "images": paths, "label": "nonlens"})

    rng = np.random.RandomState(SEED)
    rng.shuffle(out)
    p = OUT / "sft_v5r3_train.jsonl"
    with open(p, "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    labs = pd.Series([x["label"] for x in out]).value_counts().to_dict()
    tgts = len({x["messages"][2]["content"] for x in out})
    print(f"[v5r3] {len(out)} rows ({labs}) | unique targets {tgts} | rf missing {miss} -> {p}")


if __name__ == "__main__":
    main()
