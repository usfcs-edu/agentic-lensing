#!/usr/bin/env python
"""gen_p2c_manifest.py -- emit the P2c cell manifest (single source of truth).

Prints one line per job; the P2c slurm scripts lane-schedule them across the
4-A100 node. Line format:

    <kind> <gpu_dtype> <arg...>

  kind=cell   -> `python 22_run_cell.py <arg...>`   (T2/T3 eval matrix)
  kind=script -> `python <arg...>`                   (23 PT ref / 24 SMC evid)

gpu_dtype (f32|f64) is informational for the scheduler log only -- 22_run_cell
sets x64 per process from the registry, so mixed dtypes co-exist on one node.

BUDGETS below are the smoke-calibrated Track-A budgets (see CAMPAIGN.md P2c
launch plan); Track-B is `--track B` (22_run_cell scales the adapter SCALE_KEYS
x4 automatically). Edit the constants, re-run, review the manifest.

  --kind matrix     : the 24 T2/T3 eval cells (default)
  --kind reference  : the 23 PT-reference + 24 SMC-evidence jobs (6 jobs)
  --kind all        : both
  --seeds 0,1,2     : seed list (default 0,1,2)
  --no-glnt         : drop the T3 glnt cells (lowest priority, time-tight)
"""
from __future__ import annotations

import argparse
import json

T2 = "foundry_marg46"    # 46-dim ridge-marg, cond-1e14, f64
T3 = "foundry_v3b74"     # 74-dim bimodal, f32

# ---- Track-A budgets (smoke-calibrated; Track-B = these with SCALE_KEYS x4) --
# T2 (f64, cond-1e14): s0 owns it (precond_fixedL diagraw); bj_mclmc = the
# auto SVI-diag "no hand-built matrix" challenger.
T2_S0 = dict(n_chains=24, burn=1000, keep=4000)                 # SCALE_KEY n_keep
T2_MCLMC = dict(budget=dict(n_chains=8, n_tune=2000, n_steps=12000))  # SK n_steps

# T3 (f32, bimodal): judged on within-mode ESS + mode sensitivity (NOT R-hat).
# s0 chees on T3 runs a FRESH MAP+SVI; guards.check_svi_schedule forces
# >=150/dim = 11100 SVI steps, so n_vi dominates the cost. n_vi=64 (down from
# the gu-2022 200) keeps the 74-dim SVI feasible (~0.7 A100-h) while staying a
# valid ELBO estimator. glnt/remc/bj_mclmc reuse the STORED svi_v12_v3br via
# gaussian_init and pay none of this. (documented P2c deviation)
T3_S0 = dict(n_chains=16, burn=300, keep=1000,                  # reference recipe
             config=dict(n_map=16, map_steps=150, n_vi=64))     # (mode-collapse
#                        ref, not a money method -> lean; consider 1 seed)
# glnt's stage-2 SMC anneal is the SAME 145 MB/particle memory bound as 24, so
# n_particles=300 (default 1000 OOMs even on A100). anneal_init="svi" because
# the prior->posterior path was infeasible on T1 (lambda stalls ~0.2; P2b), and
# is worse on the v3b bimodal barrier. recipe-under-test; drop-first if tight.
T3_GLNT = dict(budget=dict(n_chains=48, n_burn=300, n_keep=1500,
                           n_particles=300),
               config=dict(anneal_init="svi", max_lambda_steps=120))
# remc_pt RE-TUNED beta-ladder for T3 (blew up on cond-1e14; T3 basins are
# Delta chi^2_v ~ 0.08 close -> beta_min ~ 0.05, 8 replicas; verify swaps).
T3_REMC = dict(budget=dict(n_chains=8, n_burn=500, n_keep=2500),
               config=dict(n_replicas=8, beta_min=0.02, num_leapfrog=8,
                           eps0=0.15, eps_scale_power=-0.5, mass="svi_diag"))
# nautilus on 74-dim (self-checked unit-cube face): efficiency + mode recovery.
T3_NAUT = dict(budget=dict(n_eff=2000, n_like_max=6_000_000),
               config=dict(n_live=2000, n_networks=4, n_batch=512))

# ---- PT reference + SMC evidence (the T3 mode-weight reference) --------------
# eps0/eps_max are the phoenix-smoke healthiest values (eps0=0.15 froze the
# inner HMC -> fake 100% swaps; eps0=0.03/eps_max=0.3 -> swaps ~0.64 + real
# exploration). NOTE: NO smoke config achieved basin MIXING (the likelihood-
# only hot end doesn't bridge the tight gamma prior); this PT_REF is a STARTING
# point -- run the beta_min/eps tuning sweep (P2c launch plan) first, and treat
# 24's basin-SMC weights as PRIMARY, PT as the cross-check.
PT_REF = dict(n_replicas=10, beta_min=0.02, n_chains=8, n_burn=1000,
              max_steps=12000, num_leapfrog=8, eps0=0.03, eps_max=0.3,
              mass="svi_diag", target_round_trips=100)
# n_particles is MEMORY-BOUND on v3b: the particle-with-gradient SMC costs
# ~145 MB/particle (260x260 supersampled sim + 28 shapelet-amp AD tape; vs
# ~21 MB/particle on T1). 300 particles ~= 43 GB, fits an A100 80 GB with
# margin; n_particles >~ 500 OOMs. More repeats compensate for fewer particles.
SMC_EVID = dict(n_particles=300, n_repeats=5, num_mcmc_steps=4,
                hmc_integration_steps=8, hmc_step_size=0.1, cov_inflate=2.0)


def cell(sampler, target, seed, track, extra):
    dt = "f64" if target == T2 else "f32"
    parts = ["cell", dt, "--sampler", sampler, "--target", target,
             "--seed", str(seed), "--track", track, *extra]
    return " ".join(parts)


def budget_args(b):
    out = []
    for flag, key in (("--n-chains", "n_chains"), ("--burn", "burn"),
                      ("--keep", "keep")):
        if key in b:
            out += [flag, str(b[key])]
    if "budget" in b:
        out += ["--budget-json", json.dumps(b["budget"], separators=(",", ":"))]
    if "config" in b:
        out += ["--config-json", json.dumps(b["config"], separators=(",", ":"))]
    return out


def matrix_cells(seeds, with_glnt):
    lines = []
    for s in seeds:                                   # T2: 2 methods x {A,B}
        for track in ("A", "B"):
            lines.append(cell("s0_baseline", T2, s, track, budget_args(T2_S0)))
            lines.append(cell("bj_mclmc", T2, s, track, budget_args(T2_MCLMC)))
    for s in seeds:                                   # T3: Track A only
        lines.append(cell("s0_baseline", T3, s, "A", budget_args(T3_S0)))
        lines.append(cell("remc_pt", T3, s, "A", budget_args(T3_REMC)))
        lines.append(cell("nautilus_runner", T3, s, "A", budget_args(T3_NAUT)))
        if with_glnt:
            lines.append(cell("glnt", T3, s, "A", budget_args(T3_GLNT)))
    return lines


def reference_jobs(seeds):
    lines = []
    for s in seeds:
        pt = ["script", "f32", "23_pt_reference_v3b.py", "--seed", str(s),
              "--n-replicas", str(PT_REF["n_replicas"]),
              "--beta-min", str(PT_REF["beta_min"]),
              "--n-chains", str(PT_REF["n_chains"]),
              "--n-burn", str(PT_REF["n_burn"]),
              "--max-steps", str(PT_REF["max_steps"]),
              "--num-leapfrog", str(PT_REF["num_leapfrog"]),
              "--eps0", str(PT_REF["eps0"]), "--eps-max", str(PT_REF["eps_max"]),
              "--mass", PT_REF["mass"],
              "--target-round-trips", str(PT_REF["target_round_trips"])]
        lines.append(" ".join(pt))
    for s in seeds:
        sm = ["script", "f32", "24_basin_evidence_v3b.py", "--seed", str(s),
              "--n-particles", str(SMC_EVID["n_particles"]),
              "--n-repeats", str(SMC_EVID["n_repeats"]),
              "--num-mcmc-steps", str(SMC_EVID["num_mcmc_steps"]),
              "--hmc-integration-steps", str(SMC_EVID["hmc_integration_steps"]),
              "--hmc-step-size", str(SMC_EVID["hmc_step_size"]),
              "--cov-inflate", str(SMC_EVID["cov_inflate"])]
        lines.append(" ".join(sm))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="matrix",
                    choices=["matrix", "reference", "all"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--no-glnt", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    lines = []
    if args.kind in ("matrix", "all"):
        lines += matrix_cells(seeds, not args.no_glnt)
    if args.kind in ("reference", "all"):
        lines += reference_jobs(seeds)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
