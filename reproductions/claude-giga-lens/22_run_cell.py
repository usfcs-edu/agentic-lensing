#!/usr/bin/env python
"""22_run_cell.py -- run ONE benchmark cell (sampler x target x seed x track).

Asserts the zoo freeze before running (structure here + the adapter-fidelity
logp assertion inside run_cell), then writes the cgl.io npz/json result pair
under data/results/<sampler>/<target>/.

One cell = one process (dtype isolation; GIGALENS_X64 and XLA flags are set
from the registry's STATIC dtype before jax is imported).

P2b: all 9 adapters run through this driver against the same run_cell
signature. P2b additions (all additive):
  * per-tier Track-A gradient budgets (T0 2e5 / T1 1.5e6) exported to the
    adapter as budget["n_grad_budget"] (gradient-free adapters get 2x that
    in likelihood evals per protocol);
  * --frozen: apply the (sampler, tier) policy from data/policies_frozen.json
    (config + budget); REQUIRED once the freeze file exists for any
    T0/T1 track-A/B cell (pre-registered eval discipline) -- the resolved
    config/budget hash is asserted against the freeze;
  * --track B: convergence track -- frozen config, frozen budget with the
    adapter's SCALE_KEYS fields x4 (cap; time-to-convergence is measured by
    prefix analysis in 25_pool_benchmark.py);
  * --budget-json for pilot/tuning overrides (pre-freeze only).

Run (GPU 9, the campaign L4):
  /raid/benson/.venvs/cgl/bin/python 22_run_cell.py \
      --sampler s0_baseline --target gu2022_sys000 --seed 0 --track A

Track budgets: A = the gu-2022 defaults (50 chains x 750 draws, 250 burn)
for s0_baseline; every other adapter carries its own DEFAULT_BUDGET and is
sized by its frozen policy.
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

# pre-registered Track-A gradient budgets per tier (README section P2 / P2b
# protocol); Track B = 4x cap. Informational for sizing + recorded per cell.
GRAD_BUDGETS = {"T0": int(2e5), "T1": int(1.5e6)}
TRACK_B_SCALE = 4

# the pre-registered eval split (policies frozen before these cells run)
EVAL_TARGETS = {f"gu2022_sys{i:03d}" for i in (6, 7, 8, 9, 10, 11)}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampler", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default="A",
                    choices=sorted(TRACK_BUDGETS) + ["B"])
    ap.add_argument("--gpu", default=None, help="CUDA index (default 9)")
    ap.add_argument("--n-chains", type=int, default=None)
    ap.add_argument("--burn", type=int, default=None)
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--config-json", default=None,
                    help="JSON dict of adapter config overrides")
    ap.add_argument("--budget-json", default=None,
                    help="JSON dict of adapter budget overrides "
                         "(pilot/tuning only, pre-freeze)")
    ap.add_argument("--frozen", action="store_true",
                    help="apply the frozen policy from "
                         "data/policies_frozen.json for (sampler, tier)")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="LOUD escape hatch: skip the frozen-policy "
                         "assertion (never valid for eval-split cells)")
    ap.add_argument("--allow-missing-freeze", action="store_true",
                    help="unit-test escape hatch ONLY; benchmark cells must "
                         "assert the freeze")
    ap.add_argument("--results-root", default=None,
                    help="alternate results root (policy-tuning runs write "
                         "to data/results_tuning so matrix cells stay clean)")
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
    # STACK DEFECT #3b (P2b, 2026-07-06, extends the P0 priority-fusion
    # record): jaxlib 0.6.2's XLA priority-fusion pass livelocks (~680
    # spinning threads, infinite compile) in FLOAT32 too -- the P0 "f32
    # unaffected" note only covered the minimal f64 repro, and the P0 smoke
    # suite always ran with the pass disabled at module import, so the bare
    # f32 paths were never exercised. Reproduced twice on this stack:
    # (i) blackjax MCLMC tuner on t0_mix2 (f32, L4, 683 threads);
    # (ii) neutra's flowjax-NSF pullback pipeline on gu2022_sys000 (f32,
    # L4, 685 threads). Disable the pass for the affected adapters at ANY
    # dtype BEFORE the first jax import (runtime.setup_process_env keeps a
    # pre-existing --xla_disable_hlo_passes verbatim). S0 keeps its P2a
    # environment untouched.
    import os as _os
    if args.sampler in ("bj_mclmc", "neutra", "glnt"):
        _flags = _os.environ.get("XLA_FLAGS", "")
        if "--xla_disable_hlo_passes" not in _flags:
            _os.environ["XLA_FLAGS"] = (
                _flags + " --xla_disable_hlo_passes=priority-fusion").strip()
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
    from cgl.samplers import get_default_budget, get_scale_keys  # noqa: E402
    from cgl.samplers.common import (POLICIES_FROZEN,  # noqa: E402
                                     assert_frozen_policy,
                                     load_frozen_policies)

    tier = info.tier
    track = args.track
    policies = None
    if POLICIES_FROZEN.exists():
        policies = load_frozen_policies()

    if args.sampler == "s0_baseline" and track != "B" and not args.frozen:
        budget = dict(TRACK_BUDGETS[track])
        config = dict(TRACK_CONFIGS.get(track, {}))
    else:
        # contenders carry their own DEFAULT_BUDGET; frozen policies (or
        # explicit --budget-json during tuning) size them.
        budget = {}
        config = {}

    scale_after_overrides = False   # P2c non-frozen track-B (see below)
    if args.frozen or track == "B":
        has_policy = (policies is not None
                      and args.sampler in policies.get("methods", {})
                      and tier in policies["methods"].get(args.sampler, {}))
        if has_policy:
            entry = policies["methods"][args.sampler][tier]
            config = dict(entry["config"])
            budget = dict(entry["budget"])
            if track == "B":
                for k in get_scale_keys(args.sampler):
                    if budget.get(k):
                        budget[k] = int(budget[k] * TRACK_B_SCALE)
        elif args.frozen or tier in ("T0", "T1"):
            # frozen policy is REQUIRED for T0/T1 (eval discipline) and for any
            # explicit --frozen; a missing entry is a hard error.
            print(f"ERROR: no frozen policy for {args.sampler}/{tier}"
                  + ("" if policies is not None
                     else f" ({POLICIES_FROZEN} missing)"))
            return 4
        else:
            # P2c: the hard targets (T2/T3) have NO frozen policy (policies
            # were frozen for T0/T1 in P2b). The convergence track is still
            # well defined: seed the adapter DEFAULT_BUDGET, let the explicit
            # overrides below apply, then scale the adapter SCALE_KEYS x4 (same
            # pre-registered Track-B rule, config held). Track A on T2/T3 just
            # uses the adapter default + explicit overrides (no scaling).
            budget = get_default_budget(args.sampler)
            config = {}
            scale_after_overrides = (track == "B")

    # manual overrides (pilot/tuning; the freeze assertion below catches any
    # attempt to modify an eval cell)
    if args.n_chains is not None:
        budget["n_chains"] = args.n_chains
    if args.burn is not None:
        budget["n_burn"] = args.burn
    if args.keep is not None:
        budget["n_keep"] = args.keep
    if args.budget_json:
        budget.update(json.loads(args.budget_json))
    config.update(json.loads(args.config_json) if args.config_json else {})

    # P2c non-frozen track-B (T2/T3): scale the adapter SCALE_KEYS x4 AFTER the
    # explicit budget overrides, so the convergence track is 4x the actual
    # Track-A budget the same cell ran (config held fixed).
    if scale_after_overrides:
        for k in get_scale_keys(args.sampler):
            if budget.get(k):
                budget[k] = int(budget[k] * TRACK_B_SCALE)

    budget["track"] = track
    gb = GRAD_BUDGETS.get(tier)
    if gb:
        budget["n_grad_budget"] = gb * (TRACK_B_SCALE if track == "B" else 1)

    # ---- frozen-policy assertion (pre-registered eval discipline) ----------------
    if args.target in EVAL_TARGETS and track in ("A", "B"):
        if policies is None:
            print("ERROR: eval-split cells require data/policies_frozen.json"
                  " (policy freeze first!)")
            return 4
        if args.allow_unfrozen:
            print("ERROR: --allow-unfrozen is never valid for eval-split "
                  "cells")
            return 4
    if (policies is not None and track in ("A", "B") and tier in ("T0", "T1")
            and not args.allow_unfrozen
            and args.sampler in policies.get("methods", {})
            and tier in policies["methods"][args.sampler]):
        assert_frozen_policy(policies, args.sampler, tier, config, budget,
                             track)
        print(f"frozen-policy hash OK ({args.sampler}/{tier}, track {track})",
              flush=True)
    elif args.allow_unfrozen:
        print("WARNING: running UNFROZEN (tuning/diagnostic only; not an "
              "eval cell)", flush=True)

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

    results_root = Path(args.results_root) if args.results_root else None
    npz_path, json_path = io.save_cell_result(result, root=results_root)

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
