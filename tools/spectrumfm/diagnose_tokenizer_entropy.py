#!/usr/bin/env python
"""
Diagnose why the V2 spectrum tokenizer fails as a transformer tokenizer.

The Approach-A transformer only consumes the *discrete* spectrum codes
(`spec_tok.encode(x)[0]`); V2's U-Net skip connections (returned as the 3rd
element and discarded by `tokenize_and_build`) carry reconstruction info around
the quantizer. So a V2 tokenizer can reconstruct well (val_recon=0.35) while its
discrete codes are near-degenerate — which would explain spec_acc->1.0 and
redshift never igniting (z_loss pinned at ln(1024)).

This script encodes the same batch of real DESI spectra with the V1 and V2
spectrum tokenizers and reports the discrete-code entropy / codebook usage.
Run on one L4:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
  ~/.venvs/redshifty/bin/python tools/spectrumfm/diagnose_tokenizer_entropy.py
"""
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/redshifty")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nersc"))

from src.tokenizers.spectrum import SpectrumTokenizer          # noqa: E402
from src.tokenizers.spectrum_v2 import SpectrumTokenizerV2     # noqa: E402
from dr1_dataset import DR1IndexedDataset, collate_dr1_skip_none  # noqa: E402

MANIFEST = "/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl"
V1_CKPT = "/raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt"
V2_CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v2_l4x2/best.pt"
N_SPECTRA = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tok(kind, ckpt):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    if kind == "v2":
        # Auto-detect skip/cross-attn from checkpoint keys (handles --no-skip).
        has_skip = any(k.startswith("skip_proj.") for k in sd)
        has_ca = any(k.startswith("cross_attn.") for k in sd)
        print(f"  [v2 arch: skip={has_skip} cross_attn={has_ca}]")
        tok = SpectrumTokenizerV2(use_skip_connections=has_skip,
                                  use_cross_attention=has_ca).to(device)
    else:
        tok = SpectrumTokenizer().to(device)
    tok.load_state_dict(sd)
    tok.eval()
    return tok


def code_stats(name, indices):
    """indices: (B, T) long tensor of discrete codes."""
    flat = indices.reshape(-1)
    n = flat.numel()
    # global codebook usage
    uniq, counts = torch.unique(flat, return_counts=True)
    p = counts.float() / n
    entropy_bits = float(-(p * (p.clamp_min(1e-12)).log2()).sum())
    top1 = float(counts.max()) / n
    # per-sequence diversity: mean #unique codes per spectrum, and mean run-length
    per_seq_uniq = []
    for row in indices:
        per_seq_uniq.append(int(torch.unique(row).numel()))
    per_seq_uniq = sum(per_seq_uniq) / len(per_seq_uniq)
    # fraction of adjacent-equal tokens (autocorrelation proxy)
    adj_equal = float((indices[:, 1:] == indices[:, :-1]).float().mean())
    T = indices.shape[1]
    print(f"\n=== {name} ===")
    print(f"  seq length T               : {T}")
    print(f"  distinct codes used (global): {uniq.numel()}")
    print(f"  token entropy              : {entropy_bits:.3f} bits "
          f"(max possible log2(used)={math.log2(max(uniq.numel(),1)):.2f})")
    print(f"  top-1 code fraction        : {top1:.4f}")
    print(f"  mean distinct codes / spec : {per_seq_uniq:.1f} of {T}")
    print(f"  adjacent-equal fraction    : {adj_equal:.4f}")
    return dict(distinct=int(uniq.numel()), entropy_bits=entropy_bits,
                top1=top1, per_seq_uniq=per_seq_uniq, adj_equal=adj_equal, T=T)


def main():
    print(f"[device] {device}")
    with open(MANIFEST) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    ds = DR1IndexedDataset(records[:64], max_spectra=N_SPECTRA, cache_size=8)
    batch = collate_dr1_skip_none([ds[i] for i in range(len(ds))])
    flux = batch["flux"].to(device)
    ivar = batch["ivar"].to(device)
    istd = torch.sqrt(ivar.clamp(min=1e-10))
    x = torch.stack([flux, istd], dim=1)  # (B, 2, L) — matches sequences.py
    print(f"[data] batch x shape {tuple(x.shape)}")

    out = {}
    for kind, ckpt in [("v1", V1_CKPT), ("v2", V2_CKPT)]:
        tok = load_tok(kind, ckpt)
        with torch.no_grad():
            enc = tok.encode(x)
        idx = enc[0] if isinstance(enc, (tuple, list)) else enc
        out[kind] = code_stats(f"{kind.upper()} spectrum codes", idx.long().cpu())
        del tok
        torch.cuda.empty_cache()

    print("\n=== VERDICT ===")
    v1, v2 = out["v1"], out["v2"]
    print(f"  V1 entropy {v1['entropy_bits']:.2f} bits vs V2 {v2['entropy_bits']:.2f} bits")
    print(f"  V1 distinct {v1['distinct']} vs V2 {v2['distinct']}")
    print(f"  V1 adjacent-equal {v1['adj_equal']:.3f} vs V2 {v2['adj_equal']:.3f}")
    if v2["entropy_bits"] < 0.5 * v1["entropy_bits"] or v2["adj_equal"] > 0.9:
        print("  => V2 discrete codes are DEGENERATE (low entropy / highly repetitive):")
        print("     the transformer sees near-constant input -> redshift cannot ignite.")
    else:
        print("  => V2 codes are NOT obviously degenerate; failure cause is elsewhere.")


if __name__ == "__main__":
    main()
