"""
50 -- Generative redshift posteriors from AION (task 10).

Reuses the task-1 PROVABGS objects. For each input config we run the model's
*generative* head: model(encode(modalities), target_modality=[Z]) -> tok_z
logits -> softmax = a posterior over redshift on the 1024-bin grid z in [0,6].
This reproduces the paper's qualitative result that adding modalities (phot ->
phot+spectrum) sharply contracts the redshift posterior around the truth.

We quantify it with: posterior-mean point estimate R^2 vs true z, mean posterior
std (contraction, in z units), and bin-level negative-log-likelihood at the true
redshift. We also save example posteriors for a few galaxies spanning the z range.

Outputs: data/results/task10_redshift_posterior.json + figs/task10_redshift_posteriors.png

Run: HF_HOME=... python 50_redshift_posterior.py [--variant base] [--n 2000]
"""

import argparse
import json

import numpy as np

import _config as C

RAW = C.RAW / "provabgs"
PHOT_BANDS = ["G", "R", "Z", "W1"]
FLUX_CLS = {"G": "LegacySurveyFluxG", "R": "LegacySurveyFluxR",
            "Z": "LegacySurveyFluxZ", "W1": "LegacySurveyFluxW1"}


def build_mods(M, rows, flux, spec=None):
    import torch
    mods = []
    for j, b in enumerate(PHOT_BANDS):
        cls = getattr(M, FLUX_CLS[b])
        mods.append(cls(value=torch.as_tensor(flux[rows, j:j+1].astype(np.float32),
                                               device="cuda")))
    if spec is not None:
        si = spec["row2pos"]
        pos = np.array([si[r] for r in rows])
        mods.append(M.DESISpectrum(
            flux=torch.as_tensor(spec["flux"][pos].astype(np.float32), device="cuda"),
            ivar=torch.as_tensor(spec["ivar"][pos].astype(np.float32), device="cuda"),
            mask=torch.as_tensor(spec["mask"][pos].astype(bool), device="cuda"),
            wavelength=torch.as_tensor(spec["wave"][pos].astype(np.float32), device="cuda")))
    return mods


def posterior_for(config, model, cm, M, rows, flux, spec, zgrid, batch=64):
    import torch
    post = np.zeros((len(rows), len(zgrid)), dtype=np.float32)
    use_spec = spec if "spec" in config else None
    for s in range(0, len(rows), batch):
        rb = rows[s:s+batch]
        mods = build_mods(M, rb, flux, use_spec)
        tokens = cm.encode(*mods)
        out = model(tokens, target_modality=[M.Z])
        # tok_z logits are width Z_NBINS+1 (1024 value bins + 1 sentinel);
        # restrict to the value bins and renormalise to a redshift posterior.
        logits = out["tok_z"].squeeze(1).float()[:, : len(zgrid)]
        p = torch.softmax(logits, dim=-1)  # (b, nbins)
        post[s:s+batch] = p.cpu().numpy()
    return post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    C.seed_everything()

    import torch
    import aion.modalities as M
    from aion.codecs import CodecManager
    from aion.model import AION
    torch.set_grad_enabled(False)

    flux = np.load(RAW / "flux.npy")
    zt = np.load(RAW / "targets.npy")[:, 0]  # true z (column 0)
    spec_idx = np.load(RAW / "spec_index.npy")
    spec = {
        "flux": np.load(RAW / "spec_flux.npy"), "ivar": np.load(RAW / "spec_ivar.npy"),
        "mask": np.load(RAW / "spec_mask.npy"), "wave": np.load(RAW / "spec_wave.npy"),
        "row2pos": {int(r): k for k, r in enumerate(spec_idx)},
    }
    # evaluate on the spec-overlap set so phot vs phot+spec compare on same galaxies
    rng = np.random.default_rng(C.SEED)
    pool = spec_idx.copy()
    rows = np.sort(rng.choice(pool, min(args.n, len(pool)), replace=False))

    cm = CodecManager(device="cuda")
    model = AION.from_pretrained(C.MODELS[args.variant]).to("cuda").eval()

    nb = C.Z_NBINS
    zgrid = (np.arange(nb) + 0.5) / nb * C.Z_RANGE[1]  # bin centers in z
    results = {}
    posts = {}
    for config in ["phot", "phot_spec"]:
        post = posterior_for(config, model, cm, M, rows, flux, spec, zgrid)
        posts[config] = post
        pmean = (post * zgrid).sum(1)
        pstd = np.sqrt((post * (zgrid[None, :] - pmean[:, None])**2).sum(1))
        ztrue = zt[rows]
        ss_res = np.sum((ztrue - pmean)**2)
        ss_tot = np.sum((ztrue - ztrue.mean())**2)
        r2 = 1 - ss_res / ss_tot
        # NLL at true bin
        tb = np.clip((ztrue / C.Z_RANGE[1] * nb).astype(int), 0, nb - 1)
        nll = -np.mean(np.log(post[np.arange(len(rows)), tb] + 1e-12))
        results[config] = {
            "point_R2": round(float(r2), 4),
            "mean_post_std": round(float(pstd.mean()), 4),
            "nll_true": round(float(nll), 4),
            "n": int(len(rows)),
        }
        print(f"[{args.variant}/{config}] z point-R2={r2:.3f} "
              f"mean_post_std={pstd.mean():.4f} NLL={nll:.3f}")

    (C.RESULTS / "task10_redshift_posterior.json").write_text(
        json.dumps({args.variant: results}, indent=2))

    # figure: example posteriors spanning z range
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ztrue = zt[rows]
    order = np.argsort(ztrue)
    picks = order[np.linspace(0, len(order)-1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, gi in zip(axes.ravel(), picks):
        for config, color in [("phot", "C0"), ("phot_spec", "C1")]:
            ax.plot(zgrid, posts[config][gi], color=color, label=config)
        ax.axvline(ztrue[gi], color="k", ls="--", alpha=0.6, label="true z")
        ax.set_xlim(max(0, ztrue[gi]-0.3), ztrue[gi]+0.3)
        ax.set_title(f"z_true={ztrue[gi]:.3f}")
        ax.set_xlabel("redshift")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"AION-{args.variant} redshift posterior: photometry vs +spectrum")
    fig.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGS / "task10_redshift_posteriors.png", dpi=120)
    print("REDSHIFT_POSTERIOR_OK")


if __name__ == "__main__":
    main()
