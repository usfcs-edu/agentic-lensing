"""Stage C: SVI at paper scale, started from the Stage-B MAP.

Full-covariance Gaussian variational posterior via the library SVI
(shard_map ELBO across devices). The learning rate ramps 0 -> lr_max
quadratically (the published GIGA-Lens recipe). Gates per the research
group: the ELBO trace must be smooth AND flattened, and the variational
covariance must be FULL RANK (min eigenvalue > 0 at float64) so the HMC
momentum preconditioner inv(cov) is well defined.

Run:
  python 42_svi_paper_scale.py --start data/map_v11_prod.npz --n-vi 500 \
      --num-steps 1500 --tag prod
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import _data_lib as D

REPRO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="data/map_v11_prod.npz")
    ap.add_argument("--n-vi", type=int, default=500)
    ap.add_argument("--num-steps", type=int, default=1500)
    ap.add_argument("--lr-max", type=float, default=1e-3)
    ap.add_argument("--init-scales", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=D.N_MAX)
    ap.add_argument("--data", type=str, default="cutout_v2.npz")
    ap.add_argument("--tag", type=str, default="prod")
    args = ap.parse_args()

    D.bootstrap_vendor()
    import jax
    import jax.experimental.shard_map  # noqa: F401
    import jax.numpy as jnp
    import optax
    from gigalens.jax.inference import ModellingSequence

    print(f"devices: {jax.devices()}")
    d, prior, phys, prob, sim_config = D.build_all(n_max=args.n_max, data_file=args.data)
    seq = ModellingSequence(phys, prob, sim_config)

    start_z = np.load(REPRO / args.start)["best_z"].astype(np.float32)
    print(f"start: {args.start} (z dim {start_z.shape})")

    sched = optax.polynomial_schedule(
        init_value=1e-7, end_value=args.lr_max, power=2.0,
        transition_steps=args.num_steps)
    opt = optax.adam(sched)

    t0 = time.time()
    qz, loss_hist = seq.SVI(
        jnp.asarray(start_z), opt, n_vi=args.n_vi,
        init_scales=args.init_scales, num_steps=args.num_steps,
        seed=args.seed, pbar_interval=0)
    elapsed = time.time() - t0

    loss_hist = np.asarray(loss_hist, dtype=np.float64)
    loc = np.asarray(qz.loc, dtype=np.float64)
    cov = np.asarray(qz.covariance(), dtype=np.float64)

    # ---- gates -------------------------------------------------------------
    k = max(1, args.num_steps // 10)
    tail, prev = loss_hist[-k:], loss_hist[-2 * k:-k]
    flat_frac = float((prev.mean() - tail.mean()) / max(abs(tail.mean()), 1e-9))
    diffs = np.diff(loss_hist[-2 * k:])
    smooth = bool(np.all(np.isfinite(loss_hist)) and
                  np.abs(diffs - diffs.mean()).max() < 12 * (diffs.std() + 1e-12))
    flat = bool(abs(flat_frac) < 0.005)

    eigs = np.linalg.eigvalsh(cov)
    min_eig, max_eig = float(eigs[0]), float(eigs[-1])
    full_rank = bool(min_eig > 0)
    cond = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    rank = int((eigs > 1e-12 * max_eig).sum())

    gate_ok = flat and smooth and full_rank
    out = dict(
        tag=args.tag, start=args.start, n_vi=args.n_vi,
        num_steps=args.num_steps, lr_max=args.lr_max,
        init_scales=args.init_scales, seed=args.seed,
        elapsed_s=elapsed, final_elbo=float(loss_hist[-1]),
        best_elbo=float(np.nanmin(loss_hist)),
        flat_frac=flat_frac, gate_elbo_flat=flat, gate_elbo_smooth=smooth,
        min_eig=min_eig, max_eig=max_eig, cond=cond,
        rank=rank, dim=int(loc.size), gate_full_rank=full_rank,
        gate_ok=bool(gate_ok),
    )
    np.savez(REPRO / "data" / f"svi_v12_{args.tag}.npz",
             loc=loc, cov=cov,
             scale_tril=np.asarray(qz.scale_tril, dtype=np.float64)
             if hasattr(qz, "scale_tril") else np.linalg.cholesky(cov),
             loss_hist=loss_hist, meta=json.dumps(out))
    (REPRO / "data" / f"svi_v12_{args.tag}_gate.json").write_text(
        json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nSVI [{args.tag}]: {elapsed:.0f}s  ELBO {loss_hist[0]:.1f} -> "
          f"{loss_hist[-1]:.1f}  rank {rank}/{loc.size}  "
          f"-> GATE {'PASS' if gate_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
