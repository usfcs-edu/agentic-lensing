#!/usr/bin/env python3
"""finetune/wise_ft.py — WiSE-FT weight-space interpolation (Wortsman+22) for the v5 students.

Builds `alpha * student + (1 - alpha) * base` shard-by-shard from safetensors (no model class
needed, ~one shard of RAM), mirroring the student's shard layout; config/tokenizer/processor
files are copied from the student directory. The base may be a local dir or an HF-cache id
(resolved offline).

Rationale: the fine-tuned student and the zero-shot base have complementary error profiles on
bench v5.1 (student: recovery 68% vs 47%; base: near-miss rejection 74% vs 58%) — WiSE-FT's
classic result is that intermediate alpha keeps most of the fine-tune gain while recovering
base robustness.

  python -m lensjudge.finetune.wise_ft --base Qwen/Qwen3.5-27B \
      --student /path/checkpoint-600-merged --alpha 0.7 --out /path/wise_a70
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _resolve(path_or_id: str) -> Path:
    p = Path(path_or_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path_or_id, local_files_only=True))


def _key_map(model_dir: Path) -> dict[str, Path]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: model_dir / v for k, v in wm.items()}
    single = model_dir / "model.safetensors"
    with safe_open(single, framework="pt") as f:
        return {k: single for k in f.keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="student weight (1.0 = pure student)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base_dir, stu_dir, out = _resolve(a.base), Path(a.student), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base_map = _key_map(base_dir)
    stu_map = _key_map(stu_dir)
    missing = set(stu_map) ^ set(base_map)
    assert not missing, f"key sets differ ({len(missing)}), e.g. {sorted(missing)[:5]}"

    # group student keys by their shard so each shard is written once, layout preserved
    by_shard: dict[Path, list[str]] = {}
    for k, shard in stu_map.items():
        by_shard.setdefault(shard, []).append(k)

    base_handles: dict[Path, object] = {}
    for shard, keys in sorted(by_shard.items()):
        merged = {}
        with safe_open(shard, framework="pt") as fs:
            for k in keys:
                ts = fs.get_tensor(k)
                bshard = base_map[k]
                if bshard not in base_handles:
                    base_handles[bshard] = safe_open(bshard, framework="pt")
                tb = base_handles[bshard].get_tensor(k)
                if ts.is_floating_point():
                    merged[k] = (a.alpha * ts.float() + (1 - a.alpha) * tb.float()).to(ts.dtype)
                else:
                    merged[k] = ts
        save_file(merged, out / shard.name, metadata={"format": "pt"})
        print(f"[wise] wrote {shard.name} ({len(merged)} tensors)")

    for f in stu_dir.iterdir():
        if f.suffix in {".json", ".txt", ".jinja"} or f.name.startswith(("tokenizer", "vocab", "merges")):
            shutil.copy2(f, out / f.name)
    print(f"[wise] alpha={a.alpha} -> {out}")


if __name__ == "__main__":
    main()
