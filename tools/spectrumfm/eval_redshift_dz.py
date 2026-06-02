#!/usr/bin/env python
"""
Binning-fair redshift comparison across Approach-A checkpoints.

The raw `redshift_acc` (exact-bin match) is NOT comparable between V1 (256-level
RedshiftTokenizer) and V2 (1024-level RedshiftTokenizerV2). This script computes
a binning-independent, physical metric: it masks the redshift token in the
ENCODER (so the model must predict z from the spectrum, not copy it), reads the
decoder's redshift prediction at position 0, decodes it to a continuous z via the
run's own redshift tokenizer, and compares to the TRUE continuous z with
dz/(1+z). Reports the standard DESI-style tolerances.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
  ~/.venvs/redshifty/bin/python tools/spectrumfm/eval_redshift_dz.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/redshifty")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nersc"))

from src.models.transformer import (  # noqa: E402
    SpectrumTransformer, TOTAL_VOCAB_SIZE, MASK_TOKEN, REDSHIFT_TOKEN_OFFSET,
)
from src.tokenizers.spectrum import SpectrumTokenizer            # noqa: E402
from src.tokenizers.spectrum_v2 import SpectrumTokenizerV2       # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer            # noqa: E402
from src.tokenizers.redshift_v2 import RedshiftTokenizerV2       # noqa: E402
from src.training.sequences import tokenize_and_build            # noqa: E402
from dr1_dataset import DR1IndexedDataset, collate_dr1_skip_none  # noqa: E402

MANIFEST = "/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl"
CKPT_DIR = Path("/raid/benson/data/desi_dr1_medium/checkpoints/checkpoints")
N_SPECTRA = 512
device = torch.device("cuda")

RUNS = [
    ("V1 (ConvNeXt+LFQ, 256-lvl z)",      CKPT_DIR / "approach_a_mix_l4x2_v1" / "best.pt"),
    ("V2 no-skip (1024-lvl z)",           CKPT_DIR / "approach_a_mix_l4x2_v2noskip" / "best.pt"),
]


def build_spec_tok(path):
    sd = torch.load(path, map_location=device, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    # V1 and V2 both have 'decoder_stages.*' keys, so discriminate by the
    # checkpoint directory name (tokenizer_v1_* vs tokenizer_v2*).
    is_v2 = "v2" in Path(path).parent.name.lower()
    if is_v2:
        has_skip = any(k.startswith("skip_proj.") for k in sd)
        has_ca = any(k.startswith("cross_attn.") for k in sd)
        tok = SpectrumTokenizerV2(use_skip_connections=has_skip, use_cross_attention=has_ca)
    else:
        tok = SpectrumTokenizer()
    tok = tok.to(device)
    tok.load_state_dict(sd)
    tok.eval()
    return tok


def build_z_tok(z_state):
    n = int(z_state["n_levels"])
    cls = RedshiftTokenizerV2 if n == 1024 else RedshiftTokenizer
    zt = cls(n_levels=n, gaussian_range=float(z_state["gaussian_range"]))
    zt._sorted_z = z_state["sorted_z"].cpu()
    zt._min_z = float(zt._sorted_z[0]); zt._max_z = float(zt._sorted_z[-1])
    # RedshiftTokenizerV2.is_fitted also requires _embedding (unused by
    # encode/decode); set a dummy so the is_fitted gate passes.
    if isinstance(zt, RedshiftTokenizerV2):
        zt._embedding = nn.Linear(zt.n_levels, zt.d_model, bias=False)
    return zt


def evaluate_dz(name, ckpt_path, batch):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    approach = ck.get("approach", "a")
    assert approach == "a", f"this readout assumes approach a (rz at encoder pos 1), got {approach}"
    z_tok = build_z_tok(ck["z_tokenizer"])
    spec_tok = build_spec_tok(Path(ck["tokenizer_ckpt_path"]))
    model = SpectrumTransformer(vocab_size=TOTAL_VOCAB_SIZE, d_model=768,
                                n_encoder_layers=6, n_decoder_layers=6, n_heads=12).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        enc, dec, tgt, _, _ = tokenize_and_build(
            batch, spec_tok, z_tok, "a", device, encoder_mask_ratio=0.0)
        # Honest readout: hide the true redshift from the ENCODER (pos 1 = rz for approach a),
        # leaving the spectrum visible, so the model must predict z from the spectrum.
        enc = enc.clone(); enc[:, 1] = MASK_TOKEN
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(enc, dec)
        n_lvl = z_tok.n_levels
        rs_logits = logits[:, 0, REDSHIFT_TOKEN_OFFSET:REDSHIFT_TOKEN_OFFSET + n_lvl].float()
        pred_bin = rs_logits.argmax(-1).cpu()
    z_pred = z_tok.decode(pred_bin)
    z_true = batch["z"].cpu().flatten()
    dz = (z_pred - z_true).abs() / (1.0 + z_true)
    out = dict(
        name=name, n=int(z_true.numel()), n_levels=n_lvl,
        median_dz=float(dz.median()),
        within_0p0033=float((dz < 0.0033).float().mean()),   # DESI "good redshift"
        within_0p01=float((dz < 0.01).float().mean()),
        within_0p05=float((dz < 0.05).float().mean()),
        catastrophic=float((dz >= 0.0033).float().mean()),
    )
    return out


def main():
    print(f"[device] {device}")
    with open(MANIFEST) as fh:
        import json
        records = [json.loads(line) for line in fh if line.strip()]
    ds = DR1IndexedDataset(records[:96], max_spectra=N_SPECTRA, cache_size=12)
    batch = collate_dr1_skip_none([ds[i] for i in range(len(ds))])
    print(f"[data] {batch['z'].numel()} spectra; z range "
          f"[{batch['z'].min():.3f}, {batch['z'].max():.3f}]")

    results = [evaluate_dz(n, p, batch) for n, p in RUNS]
    print("\n=== Binning-fair redshift prediction (encoder redshift MASKED) ===")
    hdr = f"{'run':<32} {'z-bins':>7} {'med |dz|/(1+z)':>15} {'<0.0033':>9} {'<0.01':>7} {'<0.05':>7}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<32} {r['n_levels']:>7} {r['median_dz']:>15.5f} "
              f"{r['within_0p0033']:>9.3f} {r['within_0p01']:>7.3f} {r['within_0p05']:>7.3f}")
    print("\n(<X = fraction of spectra with |dz|/(1+z) < X; 0.0033 = DESI good-z threshold)")


if __name__ == "__main__":
    main()
