#!/usr/bin/env python
"""
Codebook-health gate for the codecs (Mamba3 + RFSQ) tokenizer.

Run BEFORE wiring codecs into the Approach-A transformer. The V2 episode showed a
tokenizer can reconstruct well yet be useless downstream if its DISCRETE codes
are degenerate. This gate encodes real DESI spectra (codecs' own HDF5 cache, in
distribution) and reports RFSQ code utilization per layer + the codecs normalized
perplexity. Reference points: V1 healthy = 5.0 bits / 1024 = 0.5 normalized;
V2+skips collapsed = 0.0.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
  ~/.venvs/codecs/bin/python tools/spectrumfm/codecs_perplexity_gate.py
"""
import math
import sys
from pathlib import Path

import torch

CODECS_REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/codecs")
sys.path.insert(0, str(CODECS_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.desi import DesiDataset              # noqa: E402
from codecs_adapter import (  # noqa: E402
    load_codec_from_config, CodecsTokenizerShim, fsq_codes_to_indices)

CKPT = "/raid/benson/data/desi_dr1_medium/codecs_output_large/model.pt"
CONFIG = "/raid/benson/data/desi_dr1_medium/codecs_large.yaml"
CACHE = "/raid/benson/data/desi_dr1_medium/codecs_cache"
N_SPECTRA = 256
device = torch.device("cuda")


def code_entropy_bits(idx):
    flat = idx.reshape(-1)
    uniq, cnt = torch.unique(flat, return_counts=True)
    p = cnt.float() / flat.numel()
    ent = float(-(p * (p.clamp_min(1e-12)).log2()).sum())
    return uniq.numel(), ent, float(cnt.max()) / flat.numel()


def main():
    print(f"[device] {device}")
    codec = load_codec_from_config(CKPT, CONFIG, device=device)
    shim = CodecsTokenizerShim(codec, code_mode="layer0")
    print(f"[codec] RFSQ layer codebook sizes {shim.layer_sizes} "
          f"(compound {math.prod(shim.layer_sizes):,})")

    ds = DesiDataset(Path(CACHE))
    print(f"[data] {len(ds):,} spectra in cache; grid L={ds.wave.shape[0]}")
    batch = torch.utils.data.default_collate([ds[i] for i in range(N_SPECTRA)])
    flux = batch["flux"].to(device, dtype=torch.float32)
    ivar = batch["ivar"].to(device, dtype=torch.float32)
    x = CodecsTokenizerShim.codecs_normalize(flux, ivar)  # pre-normalized codecs input

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, codes, _ = codec.encode(x)
        layer_idx = [fsq_codes_to_indices(l.fsq, c)
                     for l, c in zip(codec.rfsq.layers, codes)]
        norm_perp = codec.perplexity(codes)
    T = layer_idx[0].shape[1]
    n_samples = layer_idx[0].numel()

    print(f"\n=== codecs RFSQ codebook health "
          f"(n={N_SPECTRA} spectra x T={T} = {n_samples:,} positions) ===")
    print(f"codecs normalized perplexity (0=collapsed, 1=uniform): {norm_perp:.3f}")
    print(f"{'layer':<6}{'size':>7}{'distinct':>10}{'entropy(bits)':>15}"
          f"{'norm-ent':>10}{'top1':>8}")
    for i, (li, sz) in enumerate(zip(layer_idx, shim.layer_sizes)):
        u, ent, top1 = code_entropy_bits(li)
        print(f"{i:<6}{sz:>7}{u:>10}{ent:>15.2f}{ent/math.log2(sz):>10.3f}{top1:>8.3f}")

    # compound (all-layer) index utilization
    comp = torch.zeros_like(layer_idx[0]); base = 1
    for li, sz in zip(layer_idx, shim.layer_sizes):
        comp = comp + li * base; base *= sz
    uc, entc, top1c = code_entropy_bits(comp)
    print(f"compound  size {base:>10,}  distinct {uc:>6}  "
          f"entropy {entc:.2f} bits  top1 {top1c:.3f}")

    u0 = code_entropy_bits(layer_idx[0])[0]
    print("\n=== VERDICT ===")
    if norm_perp > 0.10 and u0 > 4:
        print(f"  HEALTHY — codes are used (norm perplexity {norm_perp:.3f}; "
              f"layer0 uses {u0}/{shim.layer_sizes[0]} codes; compound {uc:,} distinct). "
              f"Worth a transformer run (layer0 mode fits redshifty's 1024-code slot).")
    else:
        print(f"  COLLAPSED-ish — norm perplexity {norm_perp:.3f}, layer0 {u0} codes. "
              f"Do NOT spend a transformer run; fix the tokenizer first.")


if __name__ == "__main__":
    main()
