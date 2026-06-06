"""
51 -- Spectral super-resolution / cross-modal generation (task 11).

Generates a high-resolution DESI spectrum from the low-resolution Gaia XP
coefficients of the same star, using AION's generative head:
  tokens = encode(GaiaXpBp, GaiaXpRp)
  logits = model(tokens, target_modality=DESISpectrum)
  pred   = decode(argmax(logits), DESISpectrum, wavelength=...)
(mirrors notebooks/StellarTutorial.ipynb). Compares to the true DESI spectrum.

Quantifies with median Pearson correlation between predicted and true flux over
the matched stars, and saves an example panel of predicted vs true spectra.

Outputs: data/results/task11_superres.json + figs/task11_spectral_superres.png

Run: HF_HOME=... python 51_spectral_superres.py [--variant base] [--n 200]
"""

import argparse
import json

import numpy as np

import _config as C

RAW = C.RAW / "gaia_desi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    C.seed_everything()

    import torch
    from aion.codecs import CodecManager
    from aion.model import AION
    from aion.modalities import GaiaXpBp, GaiaXpRp, DESISpectrum
    torch.set_grad_enabled(False)

    bp = np.load(RAW / "xp_bp.npy"); rp = np.load(RAW / "xp_rp.npy")
    flux = np.load(RAW / "spec_flux.npy"); mask = np.load(RAW / "spec_mask.npy")
    wave = np.load(RAW / "spec_wave.npy")
    n = min(args.n, len(bp))
    rng = np.random.default_rng(C.SEED)
    sel = rng.choice(len(bp), n, replace=False)

    cm = CodecManager(device="cuda")
    model = AION.from_pretrained(C.MODELS[args.variant]).to("cuda").eval()
    tk = DESISpectrum.token_key

    preds = []
    bs = 32
    for s in range(0, n, bs):
        b = sel[s:s+bs]
        tokens = cm.encode(
            GaiaXpBp(torch.as_tensor(bp[b], device="cuda")),
            GaiaXpRp(torch.as_tensor(rp[b], device="cuda")))
        logits = model(tokens, target_modality=[DESISpectrum])
        pred_token = {tk: logits[tk].softmax(dim=-1).argmax(dim=-1)}
        w = torch.as_tensor(wave[b].astype(np.float32), device="cuda")
        pred = cm.decode(pred_token, DESISpectrum, wavelength=w)
        preds.append(pred.flux.cpu().numpy())
    pred_flux = np.concatenate(preds, 0)

    # correlation per star over unmasked pixels
    corrs = []
    for i, gi in enumerate(sel):
        m = ~mask[gi].astype(bool)
        if m.sum() < 50:
            continue
        a, b2 = pred_flux[i][m], flux[gi][m]
        if np.std(a) > 0 and np.std(b2) > 0:
            corrs.append(float(np.corrcoef(a, b2)[0, 1]))
    corrs = np.array(corrs)
    res = {args.variant: {"median_corr": round(float(np.median(corrs)), 4),
                          "mean_corr": round(float(np.mean(corrs)), 4),
                          "n": int(len(corrs))}}
    (C.RESULTS / "task11_superres.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.variant}] gaia->desi median corr = {np.median(corrs):.3f} (n={len(corrs)})")

    # example panel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    order = np.argsort(-corrs)
    picks = order[np.linspace(0, len(order)-1, 4).astype(int)]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    for ax, pi in zip(axes, picks):
        gi = sel[pi]; m = ~mask[gi].astype(bool)
        ax.plot(wave[gi][m], gaussian_filter1d(flux[gi][m], 2), label="DESI (true)", alpha=0.6)
        ax.plot(wave[gi][m], gaussian_filter1d(pred_flux[pi][m], 2), label="AION pred (from Gaia XP)", alpha=0.6)
        ax.set_ylabel("flux"); ax.set_title(f"corr={corrs[pi]:.3f}")
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("wavelength (Å)")
    fig.suptitle(f"AION-{args.variant} spectral super-resolution: Gaia XP -> DESI")
    fig.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGS / "task11_spectral_superres.png", dpi=120)
    print("SPECTRAL_SUPERRES_OK")


if __name__ == "__main__":
    main()
