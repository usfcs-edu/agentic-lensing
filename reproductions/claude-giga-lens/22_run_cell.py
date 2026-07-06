#!/usr/bin/env python
"""22_run_cell.py -- run ONE benchmark cell (sampler x target x seed x track).

Asserts the zoo freeze before running (structure here + the adapter-fidelity
logp assertion inside run_cell), then writes the cgl.io npz/json result pair
under data/results/<sampler>/<target>/.

One cell = one process (dtype isolation; GIGALENS_X64 and XLA flags are set
from the registry's STATIC dtype before jax is imported).

Only S0 (s0_baseline) is wired in P2a; P2b adds 7 more adapters against the
same run_cell signature.

Run (GPU 9, the campaign L4):
  /raid/benson/.venvs/cgl/bin/python 22_run_cell.py \
      --sampler s0_baseline --target gu2022_sys000 --seed 0 --track A

Track budgets: A = the gu-2022 defaults (50 chains x 750 draws, 250 burn).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parent

TRACK_BUDGETS = {
    "A": dict(n_chains=50, n_burn=250, n_keep=750),
    # A5 = same budget, stored-fit-fidelity kernel (gu-2022 02_fit_system
    # DEFAULTS: fixed L=5 PHMC, NO ChEES) -- the config 03_run_batch ran the
    # T1 reference fits with; used to reproduce the documented sys003
    # pathology. Pass --track A5 (config auto-set below).
    "A5": dict(n_chains=50, n_burn=250, n_keep=750),
    "smoke": dict(n_chains=8, n_burn=100, n_keep=100),
}
TRACK_CONFIGS = {
    "A5": dict(use_gbtla=False, init_l=5),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="A", choices=sorted(TRACK_BUDGETS))
    ap.add_argument("--gpu", default=None, help="CUDA index (default 9)")
    ap.add_argument("--n-chains", type=int, default=None)
    ap.add_argument("--burn", type=int, default=None)
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--config-json", default=None,
                    help="JSON dict of adapter config overrides")
    ap.add_argument("--allow-missing-freeze", action="store_true",
                    help="unit-test escape hatch ONLY; benchmark cells must "
                         "assert the freeze")
    return ap.parse_args()


def main():
    args = parse_args()

    # env BEFORE jax import (dtype from the STATIC registry; import-light)
    from cgl.zoo import get_target_info
    from cgl.zoo.runtime import setup_process_env

    info = get_target_info(args.target)
    if not info.available:
        print(f"target {args.target} unavailable: {info.note}")
        return 2
    setup_process_env(info.dtype, args.gpu)

    import jax  # noqa: E402
    import numpy as np  # noqa: E402

    from cgl import io  # noqa: E402
    from cgl.paths import ZOO_FREEZE  # noqa: E402
    from cgl.samplers import get_run_cell  # noqa: E402
    from cgl.zoo import get_target  # noqa: E402
    from cgl.zoo.api import sha256_labels  # noqa: E402
    from cgl.zoo.runtime import (freeze_entry_for, load_freeze,  # noqa: E402
                                 setup_jax_cache)

    setup_jax_cache(REPRO)
    print(f"devices={jax.devices()} x64={jax.config.jax_enable_x64}",
          flush=True)

    # ---- budget/config ---------------------------------------------------------
    budget = dict(TRACK_BUDGETS[args.track])
    budget["track"] = args.track
    if args.n_chains is not None:
        budget["n_chains"] = args.n_chains
    if args.burn is not None:
        budget["n_burn"] = args.burn
    if args.keep is not None:
        budget["n_keep"] = args.keep
    config = dict(TRACK_CONFIGS.get(args.track, {}))
    config.update(json.loads(args.config_json) if args.config_json else {})

    # ---- build target + assert the freeze ---------------------------------------
    t0 = time.time()
    target = get_target(args.target)
    print(f"built {target.name} (dim={target.dim}, {target.dtype}) "
          f"in {time.time()-t0:.0f}s", flush=True)

    freeze_points = None
    if ZOO_FREEZE.exists():
        entry = freeze_entry_for(load_freeze(ZOO_FREEZE), args.target)
        assert entry["dim"] == target.dim, "freeze dim mismatch"
        assert entry["dtype"] == target.dtype, "freeze dtype mismatch"
        assert entry["labels_sha256"] == sha256_labels(target.labels), \
            "freeze labels hash mismatch (bijector-leaf order drifted!)"
        freeze_points = entry["points"]
        print(f"freeze structure OK ({len(freeze_points)} logp points "
              "handed to the adapter-fidelity assertion)", flush=True)
    elif not args.allow_missing_freeze:
        print(f"ERROR: {ZOO_FREEZE} missing; run 20_build_zoo.py first "
              "(or --allow-missing-freeze for smoke tests)")
        return 3

    # ---- run --------------------------------------------------------------------
    run_cell = get_run_cell(args.sampler)
    t0 = time.time()
    result = run_cell(target, seed=args.seed, budget=budget, config=config,
                      freeze_points=freeze_points)
    wall = time.time() - t0

    # ---- mode metrics (target-level; computed here so EVERY adapter gets them)
    ref = target.reference
    if ref is not None and ref.mode_assigner is not None:
        from cgl import metrics as M
        T, C, _ = result.samples.shape
        if ref.mode_assigner["method"] == "param_threshold":
            phys = target.to_physical(result.samples.reshape(T * C, -1))
            phys = {k: np.asarray(v).reshape(T, C) for k, v in phys.items()}
            assign = M.assign_modes(ref, phys=phys)
        else:
            assign = M.assign_modes(ref, Z=result.samples)
        n_modes = len(ref.mode_labels) if ref.mode_labels else \
            int(assign.max()) + 1
        result.metrics["modes"] = dict(
            occupancy=M.mode_occupancy(assign, n_modes, ref.mode_weights),
            round_trips=M.count_mode_round_trips(assign.reshape(T, C)),
            ref_weights_trusted=bool(ref.mode_weights_trusted))

    npz_path, json_path = io.save_cell_result(result)

    # ---- report -----------------------------------------------------------------
    m = result.metrics
    ds = m["diagnostics"]["summary"]
    print(f"\n=== CELL {args.sampler} x {args.target} seed={args.seed} "
          f"track={args.track} ===", flush=True)
    print(f"samples: {result.samples.shape}  wall={wall:.0f}s "
          f"(map={result.timing.get('map_s', 0):.0f}s "
          f"svi={result.timing.get('svi_s', 0):.0f}s "
          f"hmc={result.timing.get('hmc_s', 0):.0f}s)", flush=True)
    print(f"acceptance={m['acceptance_rate']:.3f}  "
          f"final_step_size={m['final_step_size']:.4g}", flush=True)
    print(f"R-hat  mass max={ds['rhat_mass']['max']:.4f}  "
          f"all max={ds['rhat_all']['max']:.4f}", flush=True)
    print(f"ESS    mass min={ds['ess_bulk_mass']['min']:.0f} "
          f"median={ds['ess_bulk_mass']['median']:.0f}   "
          f"all min={ds['ess_bulk_all']['min']:.0f}", flush=True)
    eff = m["efficiency"]
    print(f"budget n_grad={eff['n_grad']:.3g}  "
          f"ESS/grad(mass-min)={eff['ess_per_grad_mass']:.3e}  "
          f"ESS/s={eff['ess_per_sec_mass']:.3f}  [{eff['hardware']}]",
          flush=True)
    if "modes" in m:
        occ = m["modes"]["occupancy"]
        rt = m["modes"]["round_trips"]
        print(f"modes  occupancy={['%.3f' % o for o in occ['occupancy']]} "
              f"(ref {occ.get('ref_weights')}, trusted="
              f"{m['modes']['ref_weights_trusted']})  "
              f"round_trips={rt['total_round_trips']} "
              f"migrating_chains={rt['n_migrating_chains']}/{rt['n_chains']}",
              flush=True)
    print(f"freeze_check: {result.freeze_check}", flush=True)
    print(f"wrote {npz_path}\n      {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
