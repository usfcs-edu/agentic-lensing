#!/usr/bin/env python
"""51_run_rhat_blindness.py — E2-O4: the R-hat-blindness demonstration.

Design checkpoint: research/checkpoint_e2_o4.md (frozen BEFORE any run; the O1
agent owns CAMPAIGN.md this wave — checkpoint transcribed at E2 harvest).

The experiment (framed as a service to the shared methodology — Evan's rescue
for migratory first-stage MCLMC chains is empirically the same move as our
two-stage recipe): precondition a stage-2 MCLMC EXACTLY as described — init
positions drawn from the stage-1 tail (last ~50% of draws per chain), inverse
mass from the tail's regularized covariance, svi_mean = tail mean — and show
that stage-2 convergence diagnostics cannot detect stage-1 basin selection.

Arms (both on GPU 9, sequential, v3b first):
  v3b : stage 1 = E1's migrated chains (data/mclmc_diag_v3b.npz; all 64 chains
        left the physical-basin init at gamma~1.29 and equilibrated at the
        ~1.10 shelf). Prediction: stage 2 converges CLEANLY at the shelf.
  v2d : control; stage 1 = E1's healthy chains (gamma 1.4683 quotable).
        Prediction: stage 2 converges at ~1.47 — recipe fine when stage 1
        lands physically.
Falsifiers (checkpoint, reported as loudly as a pass): v3b stage-2 fails to
converge, or returns to gamma >= 1.15.

Stage-2 sampler = byte-identical E1 machinery imported from
50_run_mclmc_diag.py (vendored full_mclmc_with_adapt_sharded, MCLMC_JIT
bypassed, VMA rebind, regularize_mass_matrix=True per E1 amendment 2).
Scale = Evan's stated scale: 16 chains, 5000 burn + 5000 draws, one seed
group per arm; R-hat/ESS over the 16 chains + a 2x8 split-half readout.
NO gamma from this script is ever quotable as science (demonstration only).

Stages:
  smoke --arm {v3b,v2d} : 8 chains x 300+100 from the tail preconditioning
                          (foreground; REQUIRED PASS) -> data/o4_{arm}_stage2_smoke.json
  run   --arm {v3b,v2d} : production 16 x (5000+5000) -> data/o4_{arm}_stage2.{npz,json}
  lane                  : chain run v3b then run v2d in child processes
                          (the detached driver; log data/o4_lane.log)

Launch (E1-attempt-3 hardened discipline):
  cd ROOT && setsid nohup env PYTHONUNBUFFERED=1 GIGALENS_X64=1 CGL2_GPU=9 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /raid/benson/.venvs/cgl2/bin/python 51_run_rhat_blindness.py lane \
    </dev/null >> data/o4_lane.log 2>&1 & disown
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
DATA = ROOT / "data"

# E1 machinery imported with the 16-chain geometry (svi_mass_matrix_weight
# follows the wrapper's own 10*n_chains rule -> 160, as in E1's v3b groups).
os.environ["MCLMC_CHAINS"] = "16"
_spec = importlib.util.spec_from_file_location("e1_mclmc", ROOT / "50_run_mclmc_diag.py")
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)

# ---- frozen O4 config (checkpoint research/checkpoint_e2_o4.md) -------------
N_CHAINS = 16
BURN = 5000
DRAWS = 5000
TAIL_FRAC = 0.5                       # last 50% of stage-1 kept draws per chain
RNG_KEY_SEED = {"v3b": 230723, "v2d": 230724}
INIT_ROW_SEED = {"v3b": 1447, "v2d": 1448}
SMOKE_CHAINS, SMOKE_BURN, SMOKE_DRAWS = 8, 300, 100
SMOKE_RNG_SEED, SMOKE_ROW_SEED = 991, 992
SMOKE_SANITY_LO, SMOKE_SANITY_HI = 0.8, 2.5   # loose sanity window (NOT E1-G2:
                                              # the v3b tail violates the band
                                              # by construction — that IS the demo)
BAND_LO, BAND_HI = e1.BASIN_LO, e1.BASIN_HI   # E1-G2 band, descriptive readout only
RHAT_REF, ESS_REF = e1.RHAT_GATE, e1.ESS_GATE  # frozen E1-G1 numbers, no new knobs
HORIZON_CAP_H = 10.0                  # watchdog max_run_h; smoke HOLDS launch if exceeded

STAGE1_REF = {  # stage-1 facts recorded in the checkpoint (readout context only)
    "v3b": dict(stage1="E1 mclmc_diag_v3b (FAIL E1-G1+G2: rhat_worst 1.379, "
                       "64/64 chains migrated out of band)",
                tail_gamma_med=1.1043, expect="converges AT the migrated shelf"),
    "v2d": dict(stage1="E1 mclmc_diag_v2d (PASS: rhat_worst 1.0031, "
                       "gamma 1.4683 [1.4343, 1.5048])",
                tail_gamma_med=1.4684, expect="control: converges at ~1.47"),
}


def _guard_gpu9():
    e1._gpu_setup()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if dev != "9":
        raise RuntimeError(f"O4 is GPU 9 ONLY (GPU 8 = O2 lane); got "
                           f"CUDA_VISIBLE_DEVICES={dev!r}")


# =========================================================================== #
# Tail preconditioning — EXACTLY the Evan-described move (stage 2 sees ONLY
# stage-1 output; the E1 warm x46 cloud is never touched)
# =========================================================================== #
def build_tail(tag):
    """Pooled stage-1 tail -> (tail, cov_reg, svi_mean, meta)."""
    from cgl2.samplers import common
    src = DATA / f"mclmc_diag_{tag}.npz"
    src_md5 = e1.md5(src)
    z = np.load(src, allow_pickle=True)
    gkeys = sorted(k for k in z.files if k.startswith("pos_g"))
    mkeys = sorted(k for k in z.files if k.startswith("mass_g"))
    mass_names = [str(s) for s in z["mass_names"]]
    gi = mass_names.index("gamma")
    tails, gtails = [], []
    for pk, mk in zip(gkeys, mkeys):
        pos = np.asarray(z[pk], dtype=np.float64)      # (C, D, 46)
        n_tail = int(round(pos.shape[1] * TAIL_FRAC))
        tails.append(pos[:, -n_tail:, :].reshape(-1, pos.shape[-1]))
        gtails.append(np.asarray(z[mk], dtype=np.float64)[:, -n_tail:, gi].ravel())
    tail = np.concatenate(tails, axis=0)               # (sum C * n_tail, 46)
    gtail = np.concatenate(gtails, axis=0)
    cov_reg = common.regularize_cov(np.cov(tail.T), float(tail.shape[0]))
    svi_mean = tail.mean(axis=0)
    meta = dict(
        source=str(src), source_md5=src_md5, groups=len(gkeys),
        tail_frac=TAIL_FRAC, tail_rows=int(tail.shape[0]),
        tail_gamma=dict(med=float(np.median(gtail)),
                        q16=float(np.quantile(gtail, 0.16)),
                        q84=float(np.quantile(gtail, 0.84)),
                        min=float(gtail.min()), max=float(gtail.max())),
        construction="last 50% of E1 kept draws per chain, pooled over all "
                     "chains+seed groups; inverse-mass = regularize_cov(tail "
                     "cov, n=tail_rows) [n/(n+5)~1: PSD hygiene on the tail's "
                     "own cov]; svi_mean = tail mean; init rows drawn from "
                     "the tail (Evan-described stage-2 preconditioning)")
    print(f"[tail:{tag}] {meta['tail_rows']} rows from {len(gkeys)} groups; "
          f"gamma med {meta['tail_gamma']['med']:.4f} "
          f"[{meta['tail_gamma']['q16']:.4f}, {meta['tail_gamma']['q84']:.4f}]",
          flush=True)
    return tail, cov_reg, svi_mean, meta


def _target(tag):
    """Parity-certified diagonal target via the E1 builder (warm cloud ignored)."""
    pm, mv, arb, logdens, _Z_unused, _warm_unused = e1._build(tag)
    return pm, arb, logdens


def _split_names(pm):
    return (list(pm.model.z_param_names),
            [n for n, _ in e1.MASS_KEYS])


def _rhat_ess(zc, mc, names):
    import arviz as az
    x = np.concatenate([zc, mc], axis=2)
    ds = az.convert_to_dataset(x)
    r = np.asarray(az.rhat(ds)["x"])
    es = np.asarray(az.ess(ds, method="bulk")["x"])
    iw, ie_ = int(np.argmax(r)), int(np.argmin(es))
    return dict(rhat_worst=float(r[iw]), rhat_worst_param=names[iw],
                ess_min=float(es[ie_]), ess_min_param=names[ie_],
                rhat=dict(zip(names, map(float, r))),
                ess=dict(zip(names, map(float, es))))


# =========================================================================== #
# Stage SMOKE (foreground, REQUIRED PASS before the lane)
# =========================================================================== #
def stage_smoke(tag):
    _guard_gpu9()
    import jax
    import jax.numpy as jnp
    pm, arb, logdens = _target(tag)
    tail, cov_reg, svi_mean, tail_meta = build_tail(tag)
    rng = np.random.default_rng(SMOKE_ROW_SEED)
    pos = tail[rng.choice(tail.shape[0], SMOKE_CHAINS, replace=False)]

    t0 = time.time()
    samples, params_final = e1._mclmc_run(
        logdens, pos, cov_reg, svi_mean, rng_seed=SMOKE_RNG_SEED,
        n_chains=SMOKE_CHAINS, burn=SMOKE_BURN, draws=SMOKE_DRAWS)
    jax.block_until_ready(samples.position)
    wall = time.time() - t0
    hist = e1._hist_np(samples)
    draws = np.asarray(samples.position)[:, -SMOKE_DRAWS:, :]
    gam = arb.gamma_of(pm, draws.reshape(-1, draws.shape[-1]))

    rep = dict(generated_utc=e1.utcnow(), stage="smoke", arm=tag,
               checkpoint="research/checkpoint_e2_o4.md (frozen pre-run)",
               device=str(jax.local_devices()[0]),
               CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
               n_chains=SMOKE_CHAINS, burn=SMOKE_BURN, draws=SMOKE_DRAWS,
               tail=tail_meta, stage1_ref=STAGE1_REF[tag],
               wall_s=round(wall, 1),
               finite_draws=bool(np.all(np.isfinite(draws))),
               nonan_frac_total=float(np.mean(hist["nonan"])),
               kernel_nonan_frac=float(np.mean(hist["kernel_nonan"])),
               step_size_final_med=float(np.median(hist["step_size"][:, -1])),
               L_final=float(np.asarray(params_final.L)),
               gamma_draws=dict(med=float(np.median(gam)),
                                min=float(gam.min()), max=float(gam.max())),
               sanity_window=[SMOKE_SANITY_LO, SMOKE_SANITY_HI],
               gamma_in_sanity=bool(np.all((gam > SMOKE_SANITY_LO)
                                           & (gam < SMOKE_SANITY_HI))),
               peak_mb_smoke=round(e1._gpu_peak_mb(), 1))

    # horizon probe at production chain count (E1 pattern: 2 grads/step, x1.2)
    vg = jax.jit(jax.vmap(jax.value_and_grad(logdens)))
    zp = jnp.asarray(tail[rng.choice(tail.shape[0], N_CHAINS, replace=False)])
    v, g = vg(zp); jax.block_until_ready(v)
    t0 = time.time()
    for _ in range(10):
        v, g = vg(zp)
    jax.block_until_ready(v)
    t_batch = (time.time() - t0) / 10
    proj_h = (BURN + DRAWS) * 2 * t_batch * 1.2 / 3600.0
    rep["grad_probe"] = dict(n_chains=N_CHAINS, t_batch_s=round(t_batch, 4),
                             grads_per_step=2, margin=1.2)
    rep["projected_hours_arm"] = round(proj_h, 2)
    rep["peak_mb_with_probe"] = round(e1._gpu_peak_mb(), 1)

    rep["smoke_ok"] = bool(rep["finite_draws"]
                           and rep["kernel_nonan_frac"] >= e1.SMOKE_KERNEL_NONAN_GATE
                           and rep["nonan_frac_total"] >= 0.99
                           and rep["gamma_in_sanity"])
    out = DATA / f"o4_{tag}_stage2_smoke.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("tail", "stage1_ref")}, indent=2))
    print(f"[smoke:{tag}] {'OK' if rep['smoke_ok'] else 'FAIL'} -> {out}",
          flush=True)
    if not rep["smoke_ok"]:
        sys.exit(1)


# =========================================================================== #
# Stage RUN (production, one arm; called by the lane)
# =========================================================================== #
def stage_run(tag):
    _guard_gpu9()
    import jax
    smoke = json.loads((DATA / f"o4_{tag}_stage2_smoke.json").read_text())
    if not smoke.get("smoke_ok"):
        raise RuntimeError(f"smoke gates not passed for {tag} — refusing to run")
    pm, arb, logdens = _target(tag)
    tail, cov_reg, svi_mean, tail_meta = build_tail(tag)
    rng = np.random.default_rng(INIT_ROW_SEED[tag])
    row_idx = rng.choice(tail.shape[0], N_CHAINS, replace=False)
    pos = tail[row_idx]

    print(f"[run:{tag}] stage-2: {N_CHAINS} chains, burn {BURN} + draws {DRAWS}, "
          f"rng_key {RNG_KEY_SEED[tag]}, init-row seed {INIT_ROW_SEED[tag]}",
          flush=True)
    t0 = time.time()
    samples, params_final = e1._mclmc_run(
        logdens, pos, cov_reg, svi_mean, rng_seed=RNG_KEY_SEED[tag],
        n_chains=N_CHAINS, burn=BURN, draws=DRAWS)
    jax.block_until_ready(samples.position)
    wall = time.time() - t0
    hist = e1._hist_np(samples)
    kept = np.asarray(samples.position)[:, -DRAWS:, :]        # (16, 5000, 46)
    imm_final = np.asarray(samples.inverse_mass_matrix[0, -1])
    del samples
    print(f"[run:{tag}] sampling done in {wall / 3600.0:.2f} h; "
          f"nonan {np.mean(hist['nonan']):.4f}; "
          f"peak {e1._gpu_peak_mb():.0f} MB", flush=True)

    z_names, mass_names = _split_names(pm)
    names = z_names + [f"mass:{n}" for n in mass_names]
    mass = e1._mass_of(pm, kept.reshape(-1, kept.shape[-1])
                       ).reshape(N_CHAINS, DRAWS, 8)
    gam = mass[:, :, mass_names.index("gamma")]

    diag_16 = _rhat_ess(kept, mass, names)
    diag_halves = [_rhat_ess(kept[i * 8:(i + 1) * 8], mass[i * 8:(i + 1) * 8],
                             names) for i in range(2)]

    # E1-G2 band readout — DESCRIPTIVE only (report, never drop; no gate here)
    band = []
    for c in range(N_CHAINS):
        band.append(dict(chain=int(c),
                         frac_outside=float(np.mean((gam[c] <= BAND_LO)
                                                    | (gam[c] >= BAND_HI))),
                         gamma_med=float(np.median(gam[c])),
                         gamma_min=float(gam[c].min()),
                         gamma_max=float(gam[c].max())))

    qs = {str(q): float(np.quantile(gam, q))
          for q in (0.025, 0.16, 0.5, 0.84, 0.975)}
    gam_med = qs["0.5"]

    stem = DATA / f"o4_{tag}_stage2"
    npz = dict(pos=kept, mass=mass, z_names=np.array(z_names),
               mass_names=np.array(mass_names), init_pos=pos,
               init_row_idx=row_idx, cov_reg_init=cov_reg, svi_mean=svi_mean,
               imm_final=imm_final)
    for k, v in hist.items():
        npz[k] = v
    np.savez_compressed(stem.with_suffix(".npz"), **npz)

    summary = dict(
        generated_utc=e1.utcnow(), stage="run", arm=tag,
        checkpoint="research/checkpoint_e2_o4.md (frozen pre-run; verdict "
                   "language written at harvest, plots first)",
        purpose="E2-O4 R-hat-blindness demonstration — stage-2 MCLMC "
                "preconditioned on the stage-1 tail (Evan-described rescue "
                "== the two-stage recipe). DEMONSTRATION ONLY: no gamma here "
                "is a quotable science number.",
        device=str(jax.local_devices()[0]),
        CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        config=dict(
            n_chains=N_CHAINS, n_groups=1, burn=BURN, draws=DRAWS,
            frac_tune=e1.FRAC_TUNE, desired_energy_var=e1.DESIRED_ENERGY_VAR,
            num_effective_samples=e1.NUM_EFF_SAMPLES,
            svi_mass_matrix_weight=e1.SVI_MM_WEIGHT,
            windowed_mass_matrix=True, regularize_mass_matrix=e1.REGULARIZE_MM,
            step_size_adapt_use_psmile=False,
            integrator="isokinetic_mclachlan_smart",
            init_L="sqrt(46)", init_step_size="0.25*sqrt(46)",
            inverse_mass_init="regularize_cov(pooled stage-1 tail cov, "
                              "n=tail_rows)",
            rng_key_seed=RNG_KEY_SEED[tag], init_row_seed=INIT_ROW_SEED[tag],
            vma_check_disabled=True,
            sampler="vendored gigalens-linus experimental MCLMC "
                    "(full_mclmc_with_adapt_sharded, UNPATCHED on disk, "
                    "runtime _shard_map check_vma=False rebind, MCLMC_JIT "
                    "bypassed) — byte-identical E1 machinery via "
                    "50_run_mclmc_diag.py import"),
        provenance=dict(
            stage1_tail=tail_meta, stage1_ref=STAGE1_REF[tag],
            parity_refs_md5=e1.md5(DATA / "parity_refs.npz"),
            script="51_run_rhat_blindness.py"),
        wall_h=round(wall / 3600.0, 3),
        peak_mb=round(e1._gpu_peak_mb(), 1),
        nonan_frac=float(np.mean(hist["nonan"])),
        kernel_nonan_frac=float(np.mean(hist["kernel_nonan"])),
        L_final=float(np.asarray(params_final.L)),
        step_size_final_med=float(np.median(hist["step_size"][:, -1])),
        diagnostics=dict(pooled_16=diag_16, split_halves_8x2=diag_halves),
        gamma=dict(pooled_quantiles=qs,
                   delta_med_vs_stage1_tail=round(
                       gam_med - tail_meta["tail_gamma"]["med"], 5),
                   quotable=False),
        band_readout=dict(band=[BAND_LO, BAND_HI], per_chain=band,
                          note="descriptive only (E1-G2 band); the v3b arm "
                               "being outside the band IS the demonstration"),
        criteria_numbers=dict(
            note="mechanical readout vs the frozen E1-G1 numbers; the "
                 "checkpoint's O4-D1/D2/falsifier VERDICTS are written at "
                 "harvest, plots first",
            rhat_ref=RHAT_REF, ess_ref=ESS_REF,
            rhat_worst=diag_16["rhat_worst"], ess_min=diag_16["ess_min"],
            converged_at_ref=bool(diag_16["rhat_worst"] < RHAT_REF
                                  and diag_16["ess_min"] >= ESS_REF),
            gamma_med=gam_med),
    )
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    slim = {k: v for k, v in summary.items()
            if k not in ("diagnostics", "band_readout", "provenance")}
    slim["diagnostics_pooled_16"] = {k: diag_16[k] for k in
                                     ("rhat_worst", "rhat_worst_param",
                                      "ess_min", "ess_min_param")}
    print(json.dumps(slim, indent=2))
    print(f"[run:{tag}] COMPLETE -> {stem}.npz/.json "
          f"(wall {wall / 3600.0:.2f} h)", flush=True)


# =========================================================================== #
# Stage LANE — the detached driver: v3b then v2d, each in a child process
# (CUDA context fully released between arms)
# =========================================================================== #
def stage_lane():
    print(f"[lane] start {e1.utcnow()} pid={os.getpid()} ppid={os.getppid()}",
          flush=True)
    for tag in ("v3b", "v2d"):
        smoke = json.loads((DATA / f"o4_{tag}_stage2_smoke.json").read_text())
        if not smoke.get("smoke_ok"):
            raise RuntimeError(f"smoke not passed for {tag}")
    for tag in ("v3b", "v2d"):
        t0 = time.time()
        print(f"[lane] === arm {tag} start {e1.utcnow()} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "51_run_rhat_blindness.py"),
                        "run", "--arm", tag], cwd=str(ROOT), check=True)
        print(f"[lane] === arm {tag} done in {(time.time() - t0) / 3600.0:.2f} h "
              f"===", flush=True)
    print(f"[lane] COMPLETE {e1.utcnow()} — both artifacts written; "
          f"NO harvest here (house rule: plots first, later session)", flush=True)


# =========================================================================== #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "run", "lane"])
    ap.add_argument("--arm", choices=["v3b", "v2d"],
                    help="required for smoke/run")
    a = ap.parse_args()
    if a.stage == "lane":
        stage_lane()
    else:
        if not a.arm:
            ap.error("--arm required")
        (stage_smoke if a.stage == "smoke" else stage_run)(a.arm)
