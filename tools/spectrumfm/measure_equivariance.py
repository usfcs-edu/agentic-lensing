#!/usr/bin/env python
"""
Measure the redshift-EQUIVARIANCE response of Approach-A checkpoints (WS4a).

For each checkpoint, apply a FIXED synthetic redshift shift dz to a batch of
real spectra (resample the flux as if the object were at z+dz, via
physical_priors.redshift_shift_batch), run the honest encoder-masked aux-head
continuous E[z] before and after, and report the response slope

    slope = mean(E[z_shift] - E[z_orig]) / dz        (ideal = 1.0)

A model that has learned the spectrum->redshift mapping moves its prediction by
dz when the spectrum is shifted by dz (slope ~ 1). The reproduced V1 prototype
is ~non-responsive (slope ~ 0); the WS4a equivariance prior raises the small-dz
slope (e.g. 0.00 -> 0.35) — but that does NOT, on its own, improve absolute
good-z precision (see eval_per_class.py). The two capabilities are decoupled;
absolute precision is scale-limited.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 \
  ~/.venvs/redshifty/bin/python tools/spectrumfm/measure_equivariance.py \
    --checkpoints <ctrl>/best.pt <treatment>/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/redshifty")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nersc"))

from src.models.transformer import (  # noqa: E402
    SpectrumTransformer, TOTAL_VOCAB_SIZE, MASK_TOKEN,
)
from src.tokenizers.spectrum import SpectrumTokenizer            # noqa: E402
from src.tokenizers.redshift import RedshiftTokenizer            # noqa: E402
from src.training.sequences import tokenize_and_build            # noqa: E402
from dr1_dataset import DR1IndexedDataset, collate_dr1_skip_none  # noqa: E402
from physical_priors import build_wave_grid, redshift_shift_batch  # noqa: E402

MANIFEST = "/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl"
CKDIR = Path("/raid/benson/data/desi_dr1_medium/checkpoints/checkpoints")
DEFAULT = [
    str(CKDIR / "approach_a_mix_l4x2_v1" / "best.pt"),
    str(CKDIR / "approach_a_mix_l4x2_v1_eqprior" / "best.pt"),
]


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    zs = ck["z_tokenizer"]
    zt = RedshiftTokenizer(n_levels=int(zs["n_levels"]), gaussian_range=float(zs["gaussian_range"]))
    zt._sorted_z = zs["sorted_z"].cpu()
    zt._min_z = float(zt._sorted_z[0]); zt._max_z = float(zt._sorted_z[-1])
    raw = torch.load(ck["tokenizer_ckpt_path"], map_location=device, weights_only=False)
    # Per-stage downsampling strides persisted by pretrain_tokenizer; read only
    # when the ckpt is a dict (vs a bare state_dict). Absent key (old ckpts) ->
    # historical 32x (1,2,2,2) -> identical behavior.
    strides = raw.get("downsample_strides", (1, 2, 2, 2)) if isinstance(raw, dict) else (1, 2, 2, 2)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    st = SpectrumTokenizer(downsample_strides=tuple(strides)).to(device)
    st.load_state_dict(sd); st.eval()
    # Read model architecture from the checkpoint if persisted (ladder arms
    # with d_model!=768); fall back to the historical 768/6/6/12 defaults so
    # OLD checkpoints lacking "model_config" build byte-identically to today.
    cfg = ck.get("model_config", {})
    d_model = cfg.get("d_model", 768)
    n_encoder_layers = cfg.get("n_encoder_layers", 6)
    n_decoder_layers = cfg.get("n_decoder_layers", 6)
    n_heads = cfg.get("n_heads", 12)
    max_seq_len = cfg.get("max_seq_len", int(ck.get("max_seq_len", 1024)))
    m = SpectrumTransformer(vocab_size=TOTAL_VOCAB_SIZE, d_model=d_model,
                            n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers,
                            n_heads=n_heads, max_seq_len=max_seq_len).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    zc = zt.decode(torch.arange(zt.n_levels)).float().to(device)
    return m, st, zt, zc


def ez(m, st, zt, zc, flux, ivar, zz, device):
    enc, _, _, _, _ = tokenize_and_build(
        {"flux": flux, "ivar": ivar, "z": zz}, st, zt, "a", device, encoder_mask_ratio=0.0)
    enc = enc.clone(); enc[:, 1] = MASK_TOKEN   # honest: hide the true-z token
    return m.aux_z_expectation(enc, zc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--n-spectra", type=int, default=256)
    ap.add_argument("--dz", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    recs = [json.loads(l) for l in open(args.manifest) if l.strip()]
    ds = DR1IndexedDataset(recs[:60], max_spectra=args.n_spectra, cache_size=12)
    batch = collate_dr1_skip_none([ds[i] for i in range(len(ds))])
    wave = build_wave_grid(recs[0]["coadd"]).to(device)
    z = batch["z"].to(device).float()
    flux = batch["flux"].to(device).float(); ivar = batch["ivar"].to(device).float()
    print(f"N={z.numel()}  equivariance response d(Ez)/d(dz)  (ideal slope = 1.0)")

    for ckpt in args.checkpoints:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            print(f"[skip] missing {ckpt}"); continue
        m, st, zt, zc = load(ckpt, device)
        print(f"\n[{ckpt.parent.name}]")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ez0 = ez(m, st, zt, zc, flux, ivar, z, device)
            for d in args.dz:
                dz = torch.full_like(z, d)
                sf, si = redshift_shift_batch(flux, ivar, z, dz, wave)
                dezz = (ez(m, st, zt, zc, sf, si, z + dz, device) - ez0)
                print(f"   dz=+{d:.2f}: mean d(Ez)={float(dezz.mean()):+.4f} "
                      f"(ideal +{d:.2f}), slope={float(dezz.mean())/d:+.2f}")


if __name__ == "__main__":
    main()
