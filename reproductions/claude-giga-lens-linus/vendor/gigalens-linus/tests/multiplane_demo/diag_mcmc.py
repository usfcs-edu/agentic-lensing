"""Diagnose the HMC convergence failure (Max R-hat=1.92) from saved samples -- no re-fit.

R-hat alone says "not converged"; the trace plots say WHY (still migrating vs separated
modes vs frozen), which picks the remedy (longer chains / reparametrization / step size).
inference-diagnostics.md: winding tracks => still migrating; discrete clusters =>
multimodality; flat lines => frozen.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def main():
    d = np.load(os.path.join(ART, "demo_3plane_mcmc.npz"))
    samples = d["samples"]              # (draws, chains, params), unconstrained
    keys = list(d["keys"]); rhat = d["rhat"]; ess = d["ess"]
    short = [k.replace("planes/", "p").replace("/mass/0/", ".M.").replace("/light/0/", ".L.")
             for k in keys]

    # rank params by R-hat
    order = np.argsort(rhat)[::-1]
    print("R-hat ranking (worst first):")
    for i in order[:8]:
        print(f"  {short[i]:30s} R-hat={rhat[i]:.3f}  ESS={ess[i]:.0f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # pick: best-mixed (lowest R-hat), worst, and 2nd/3rd worst
    pick = [order[0], order[1], order[2], int(np.argmin(rhat))]
    titles = ["worst", "2nd worst", "3rd worst", "best-mixed"]
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.3))
    nchain_show = min(12, samples.shape[1])
    for a, idx, lab in zip(ax[:4], pick, titles):
        tr = samples[:, :nchain_show, idx]            # (draws, nchain_show)
        a.plot(tr, lw=0.6, alpha=0.7)
        a.set_title(f"{lab}: {short[idx]}\nR-hat={rhat[idx]:.2f} ESS={ess[idx]:.0f}")
        a.set_xlabel("draw"); a.set_ylabel("unconstrained value")
    # R-hat spectrum
    a = ax[4]
    a.bar(range(len(rhat)), np.sort(rhat)[::-1], color="indianred")
    a.axhline(1.1, color="k", ls="--", lw=1, label="accept 1.1")
    a.axhline(1.01, color="green", ls=":", lw=1, label="target 1.01")
    a.set_title("R-hat spectrum (sorted)"); a.set_xlabel("param rank"); a.set_ylabel("R-hat")
    a.legend()
    plt.tight_layout()
    path = os.path.join(ART, "demo_3plane_mcmc_diag.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
