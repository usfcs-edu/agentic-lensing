"""
25 -- Low-data regime (task 6, paper Fig 10).

Reuses the task-1 frozen embeddings. Holds a fixed test set and trains the
attentive-pooling probe on growing training subsets (N = 100..10000), tracing
R^2 vs N. The paper's point is that a frozen AION head needs very few labels to
reach most of its performance. We trace redshift (z) and stellar mass (logmass)
for the phot and phot_spec configs.

Outputs: data/results/task6_lowdata.json + figs/task6_lowdata.png

Run: python 25_lowdata.py [--variant base]
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

import _config as C
import _probe as P
from _metrics import r2_per_target

TARGETS = ["z", "logmass", "age", "logZ", "sSFR"]
TRACE = ["z", "logmass"]
NS = [100, 300, 1000, 3000, 10000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--configs", default="phot,phot_spec")
    args = ap.parse_args()
    C.seed_everything()
    v = args.variant
    targets = np.load(C.RAW / "provabgs" / "targets.npy")
    results = {}
    res_path = C.RESULTS / "task6_lowdata.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(TRACE), figsize=(11, 4.2))

    for ci, config in enumerate(args.configs.split(",")):
        emb_path = C.EMB / f"provabgs_{config}_{v}.npy"
        if not emb_path.exists():
            print(f"  [missing] {emb_path.name}")
            continue
        X = np.load(emb_path)
        idx_path = C.EMB / f"provabgs_{config}_index.npy"
        idx = np.load(idx_path) if idx_path.exists() else np.arange(len(X))
        y = targets[idx]
        # fixed test set; grow train subset from the remaining pool
        Xtr_full, Xte, ytr_full, yte = train_test_split(
            X, y, test_size=0.2, random_state=C.SEED)
        heads = {"base": 12, "large": 16, "xlarge": 32}[v]
        rng = np.random.default_rng(C.SEED)
        curve = {t: [] for t in TRACE}
        ns_used = []
        for N in NS:
            if N > len(Xtr_full):
                break
            sel = rng.choice(len(Xtr_full), N, replace=False)
            preds, r2s, _ = P.train_regression(
                Xtr_full[sel], ytr_full[sel], Xte, yte,
                lambda d, o: P.CrossAttnHead(d, o, num_heads=heads),
                epochs=120, lr=1e-3, batch_size=min(256, N), standardize_x=False,
                verbose=False)
            r2d = dict(zip(TARGETS, r2s))
            for t in TRACE:
                curve[t].append(round(float(r2d[t]), 4))
            ns_used.append(N)
            print(f"[{v}/{config}] N={N}: " +
                  " ".join(f"{t}={r2d[t]:.3f}" for t in TRACE))
        results.setdefault(config, {})[v] = {"N": ns_used, **curve}
        for ax, t in zip(axes, TRACE):
            ax.plot(ns_used, curve[t], "o-", label=config)
            ax.set_title(f"{t}: R$^2$ vs N")
            ax.set_xscale("log"); ax.set_xlabel("N train"); ax.set_ylabel("R$^2$")
        res_path.write_text(json.dumps(results, indent=2))

    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"AION-{v} low-data regime (task 6)")
    fig.tight_layout()
    C.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGS / "task6_lowdata.png", dpi=120)
    print("LOWDATA_OK ->", res_path)


if __name__ == "__main__":
    main()
