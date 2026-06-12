"""Stage B: MAP at paper scale on the v2 data product (gigalens-sean multinode-2025).

Multi-start gradient-descent MAP through the library ModellingSequence
(shard_map across all visible devices), AdaBelief with a polynomial decay
schedule (carousel-branch convention). The gate quantity is the MASKED
reduced chi^2 of the best chain (must be < 1.1 per the research group).

Run (local smoke, 2x L4):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8,9 \
    python 41_map_paper_scale.py --n-samples 64 --num-steps 100 --tag smoke
Run (Perlmutter, 4x A100):
  python 41_map_paper_scale.py --n-samples 1000 --num-steps 2000 --tag prod
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
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--num-steps", type=int, default=2000)
    ap.add_argument("--lr0", type=float, default=1e-2)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=D.N_MAX)
    ap.add_argument("--data", type=str, default="cutout_v2.npz")
    ap.add_argument("--tag", type=str, default="prod")
    ap.add_argument("--start", type=str, default=None,
                    help="warm start: npz with best_z; chains = best_z + scatter*N(0,1)")
    ap.add_argument("--scatter", type=float, default=0.05,
                    help="z-space scatter for warm starts (carousel pattern)")
    ap.add_argument("--companion3", action="store_true",
                    help="add a 3rd Sersic to the nearby companion (flexibility pass)")
    args = ap.parse_args()

    D.bootstrap_vendor()
    import jax
    import jax.experimental.shard_map  # noqa: F401 (library uses attribute path)
    import jax.numpy as jnp
    import optax
    from gigalens.jax.inference import ModellingSequence

    print(f"devices: {jax.devices()}")
    d, prior, phys, prob, sim_config = D.build_all(
        n_max=args.n_max, companion_extra=args.companion3, data_file=args.data)
    print(f"data: {d['img'].shape}, masked px: {int((~d['keep_mask']).sum())}, "
          f"noise rescale: {d['meta']['rescale']:.3f}, "
          f"n_max={args.n_max}, companion3={args.companion3}")

    seq = ModellingSequence(phys, prob, sim_config)
    sched = optax.polynomial_schedule(
        init_value=args.lr0, end_value=args.lr1, power=0.5,
        transition_steps=args.num_steps)
    opt = optax.adabelief(sched, b1=0.95, b2=0.99)

    start = None
    if args.start:
        z0 = np.load(REPRO / args.start)["best_z"].astype(np.float32)
        rng = np.random.default_rng(args.seed)
        zs = z0[None, :] + args.scatter * rng.standard_normal(
            (args.n_samples, z0.size)).astype(np.float32)
        zs[0] = z0  # keep the unperturbed incumbent in the population
        start = prob.bij.forward(list(jnp.asarray(zs).T))
        print(f"warm start from {args.start} (scatter {args.scatter})")

    t0 = time.time()
    samples, lps, chis = seq.MAP(
        opt, start=start, n_samples=args.n_samples, num_steps=args.num_steps,
        seed=args.seed, output_type="best_step", pbar_interval=0)
    elapsed = time.time() - t0

    samples = np.asarray(samples)   # (num_steps, 74) best chain per step
    lps = np.asarray(lps)
    chis = np.asarray(chis)

    best_i = int(np.nanargmax(lps))
    best_z = samples[best_i]
    best_chi = float(chis[best_i])
    final_chi = float(chis[-1])
    min_chi = float(np.nanmin(chis))

    # plateau diagnostic: relative chi^2 improvement over the last 10% of steps
    k = max(1, args.num_steps // 10)
    tail_impr = float((chis[-k] - chis[-1]) / max(chis[-1], 1e-9))

    mp = D.mass_params_from_z(prob, best_z[None, :])
    mass = {k_: float(v[0]) for k_, v in mp.items()}

    gate_ok = min_chi < 1.1
    out = dict(
        tag=args.tag, n_samples=args.n_samples, num_steps=args.num_steps,
        lr0=args.lr0, lr1=args.lr1, seed=args.seed, n_max=args.n_max,
        companion3=bool(args.companion3),
        elapsed_s=elapsed, best_step=best_i,
        best_chi2=best_chi, final_chi2=final_chi, min_chi2=min_chi,
        tail_improvement_frac=tail_impr, plateaued=bool(tail_impr < 0.01),
        gate_chi2_lt_1p1=bool(gate_ok),
        mass=mass, paper=D.PAPER,
    )
    np.savez(REPRO / "data" / f"map_v11_{args.tag}.npz",
             best_z=best_z, chi_trace=chis, lp_trace=lps,
             meta=json.dumps(out))
    (REPRO / "data" / f"map_v11_{args.tag}_gate.json").write_text(
        json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nMAP [{args.tag}]: {elapsed:.0f}s  min reduced chi2 = {min_chi:.4f} "
          f"-> GATE {'PASS' if gate_ok else 'FAIL'} (target < 1.1)")
    print(f"theta_E={mass['theta_E']:+.4f} (paper {D.PAPER['theta_E']:+.4f})  "
          f"gamma={mass['gamma']:+.4f} (paper {D.PAPER['gamma']:+.4f})")


if __name__ == "__main__":
    main()
