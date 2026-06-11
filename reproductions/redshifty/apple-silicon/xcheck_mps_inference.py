#!/usr/bin/env python3
"""xcheck_mps_inference.py — same-checkpoint MPS-vs-CUDA forward-pass fidelity (redshifty).

Loads the frozen V1 spectrum tokenizer + the Approach-A ignition transformer (step 9500,
val_loss 190.67) + the saved redshift tokenizer, builds a DETERMINISTIC teacher-forced
batch (encoder_mask_ratio=0 -> no RNG) from a fixed set of real DESI spectra, runs the
forward pass on the chosen device, and dumps a fidelity fingerprint to JSON.

Run the SAME spectra + weights on the Mac (--device mps) and on phoenix (--device cuda):
the forward kernels are the only difference, so an fp32 (--no-amp) pass must agree to a
tight tolerance. This is the redshifty analogue of the huang-2020/21 inference xcheck.

  ./xcheck_mps_inference.py --device mps  --out data/xcheck_mps.json
  # on phoenix (via its venv): --repo <redshifty> --data-root <...> --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# Fixed, diverse 4-way set of xcheck pixels (one per survey/program). Both hosts read
# these exact byte-identical FITS files in this exact order, so the model input is
# guaranteed identical and the only cross-host difference is the forward kernels.
DEFAULT_PIXELS = [
    "spectro/redux/iron/healpix/main/bright/101/10191",
    "spectro/redux/fuji/healpix/sv3/bright/104/10408",
    "spectro/redux/fuji/healpix/sv3/dark/119/11936",
    "spectro/redux/iron/healpix/main/dark/100/10048",
]


def build_records(data_root: str, pixels):
    """Build manifest-style records for an EXPLICIT ordered list of pixel rel-dirs."""
    recs = []
    for rel in pixels:
        hits = glob.glob(os.path.join(data_root, rel, "coadd-*.fits"))
        if not hits:
            continue
        coadd = hits[0]
        redrock = coadd.replace("/coadd-", "/redrock-")
        if os.path.exists(redrock):
            recs.append({"coadd": coadd, "redrock": redrock, "n_rows": -1,
                         "survey": "sv3" if "/sv3/" in coadd else "main",
                         "program": "dark" if "/dark/" in coadd else "bright"})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE / "src-redshifty"),
                    help="redshifty repo root (for src/ + nersc/ imports)")
    ap.add_argument("--ckpt-dir", default=str(
        HERE / "_raid/benson/data/desi_dr1_medium/checkpoints/checkpoints/approach_a_phase10_mix"))
    ap.add_argument("--tokenizer-ckpt", default=str(
        HERE / "_raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt"))
    ap.add_argument("--data-root", default=str(
        HERE / "_raid/benson/data/desi_dr1_medium"))
    ap.add_argument("--pixels", default=",".join(DEFAULT_PIXELS),
                    help="comma-separated pixel rel-dirs (identical set on both hosts)")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--n-spectra", type=int, default=64)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (default: fp32, the strict gate)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default=str(HERE / "data/xcheck.json"))
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "nersc"))
    import numpy as np
    import torch
    from src.models.transformer import SpectrumTransformer, TOTAL_VOCAB_SIZE
    from src.tokenizers.spectrum import SpectrumTokenizer
    from src.tokenizers.redshift import RedshiftTokenizer
    from src.training.sequences import tokenize_and_build
    from src.training.utils import compute_metrics
    from dr1_dataset import DR1IndexedDataset, collate_dr1_skip_none

    device = torch.device(a.device)
    cfg = json.loads((Path(a.ckpt_dir) / "config.json").read_text())

    # --- frozen V1 spectrum tokenizer ---
    tck = torch.load(a.tokenizer_ckpt, map_location=device, weights_only=False)
    tsd = tck.get("model", tck) if isinstance(tck, dict) else tck
    spec_tok = SpectrumTokenizer().to(device)
    spec_tok.load_state_dict(tsd)
    spec_tok.eval()
    for p in spec_tok.parameters():
        p.requires_grad_(False)

    # --- Approach-A transformer ---
    ck = torch.load(str(Path(a.ckpt_dir) / a.ckpt), map_location=device, weights_only=False)
    model = SpectrumTransformer(
        vocab_size=TOTAL_VOCAB_SIZE, d_model=cfg["d_model"],
        n_encoder_layers=cfg["n_encoder_layers"], n_decoder_layers=cfg["n_decoder_layers"],
        n_heads=cfg["n_heads"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # --- redshift tokenizer from the saved state (no redrock fit needed) ---
    zt = ck["z_tokenizer"]
    z_tok = RedshiftTokenizer(n_levels=int(zt["n_levels"]), gaussian_range=float(zt["gaussian_range"]))
    z_tok._sorted_z = zt["sorted_z"].to("cpu")
    z_tok._min_z = float(z_tok._sorted_z[0]); z_tok._max_z = float(z_tok._sorted_z[-1])

    # --- one deterministic batch of real spectra ---
    recs = build_records(a.data_root, a.pixels.split(","))
    ds = DR1IndexedDataset(recs, require_good_zwarn=True, require_nonzero_flux=True,
                           max_spectra=a.n_spectra)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.n_spectra, shuffle=False,
                                         num_workers=0, collate_fn=collate_dr1_skip_none)
    raw = next(iter(loader))
    n = int(raw["flux"].shape[0])

    with torch.no_grad():
        enc, dec, tgt, _, _ = tokenize_and_build(
            raw, spec_tok, z_tok, cfg["approach"], device, encoder_mask_ratio=0.0)
        with torch.amp.autocast(device.type, enabled=a.amp, dtype=torch.bfloat16):
            logits, loss = model(enc, dec, targets=tgt,
                                 redshift_weight=cfg["redshift_loss_weight"],
                                 aux_redshift_weight=cfg.get("aux_redshift_weight", 1.0))
        metrics = compute_metrics(logits, tgt)

    lf = logits.float().cpu().numpy().ravel()
    # strided fp64 sample of the flattened logits (prime stride) for elementwise max|Δ|
    sample = lf[::997].astype("float64").tolist()
    argmax_tokens = logits.argmax(-1).cpu().numpy().ravel().astype("int64").tolist()

    out = {
        "device": a.device, "torch": torch.__version__, "amp": a.amp,
        "ckpt": a.ckpt, "n_spectra": n,
        "loss": float(loss.item()),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "logits_shape": list(logits.shape),
        "logits_sum": float(np.float64(lf.sum())),
        "logits_absmax": float(np.abs(lf).max()),
        "logits_meanabs": float(np.abs(lf).astype("float64").mean()),
        "sample_stride": 997, "sample": sample,
        "argmax_tokens": argmax_tokens,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"[xcheck] device={a.device} amp={a.amp} n={n} loss={out['loss']:.4f} "
          f"redshift_acc={metrics['redshift_acc']:.4f} spectrum_acc={metrics['spectrum_acc']:.4f} "
          f"logits_sum={out['logits_sum']:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
