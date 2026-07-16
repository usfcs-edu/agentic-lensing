#!/usr/bin/env python
"""25a_export_v3b_basin_x46.py -- L0-G2 step 1 (OLD venv, CPU, bijector only):
export the production v3b per-basin warm-start draws in CONSTRAINED space
(canonical OLD_LABELS_46 order) so the scene-API leg can rebuild the basin
reference Gaussian q in ITS OWN unconstrained coordinates.

Why constrained space: the two stacks' unconstrained z coordinates differ
(different bijector parameterizations); constrained space is the shared,
parity-certified interface (F1-F7; the same convention as parity_refs x46).
The cross-stack q hand-off CANNOT preserve the old q exactly -- q only defines
the lambda=0 reference of the basin anneal (logZ_basin and the lambda=1
posterior are q-independent in exact arithmetic), so a Gaussian refit in
scene-z is the faithful analog. Declared in the L0-G2 design checkpoint.

Sources (the P1c/T0.2-certified production q inputs, prod_smc convention):
  low   : data/results/e2_v3b_low_canary_svicov.npz [low_draws]
  steep : data/results/e2_v3b.npz                   [steep_draws]

Mirrors 10_run_e2.prod_smc: draws = fit['<basin>_draws'] (z-space, the e2
target packing); mapping z -> x46 uses the SAME batched
cgl.likelihood._flat_named46(model.bij.forward(...)) convention as
01a_gen_parity_refs (harvest-style CPU bijector-only build, 07/08 precedent).

Run (Perlmutter login, OLD venv, CPU):
  JAX_PLATFORMS=cpu GIGALENS_X64=1 python 25a_export_v3b_basin_x46.py \
      --out /global/cfs/.../cgl2-linus/data/l0g2_basin_x46.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GIGALENS_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")

MAX_DRAWS = 4096          # per basin (subsample seed 2, rng below)
SUBSAMPLE_SEED = 2        # campaign production seed


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--low-npz", default="data/results/e2_v3b_low_canary_svicov.npz")
    ap.add_argument("--steep-npz", default="data/results/e2_v3b.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp

    from cgl.e2 import build_target

    def _flat_named46_batched(x):
        """Batched variant of cgl.likelihood._flat_named46 (verbatim traversal,
        attribution; float(...) scalar cast replaced by an (N,) array reshape
        so a whole draw matrix maps in one bijector forward)."""
        out = []
        arr = lambda v: np.asarray(v, dtype=np.float64).reshape(-1)  # noqa: E731
        for k in x[0][0]:
            out.append(("mass." + k, arr(x[0][0][k])))
        for k in x[0][1]:
            out.append(("shear." + k, arr(x[0][1][k])))
        for i, c in enumerate(x[1]):
            for k in c:
                out.append((f"LL{i}." + k, arr(c[k])))
        for k in x[2][0]:
            out.append(("srcS." + k, arr(x[2][0][k])))
        for k in x[2][1]:
            out.append(("srcShp." + k, arr(x[2][1][k])))
        return out

    # canonical order: duplicated from cgl2.param_map.OLD_LABELS_46 (this
    # script runs WITHOUT cgl2 on sys.path by design; order asserted below
    # against the model's own label set, and re-asserted scene-side)
    _MASS = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]
    _SERSIC = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y", "Ie"]
    OLD_LABELS_46 = ([f"mass.{p}" for p in _MASS]
                     + ["shear.gamma1", "shear.gamma2"]
                     + [f"LL{i}.{p}" for i in range(4) for p in _SERSIC]
                     + [f"srcS.{p}" for p in _SERSIC]
                     + [f"srcShp.{p}" for p in ["beta", "center_x", "center_y"]])
    assert len(OLD_LABELS_46) == 46

    t0 = time.time()
    target = build_target("v3b")          # CPU, bijector-only use (07/08 precedent)
    model = target.model
    assert sorted(model.index_labels) == sorted(OLD_LABELS_46), \
        "old model labels do not match the canonical 46 set"
    print(f"[25a] built v3b target (bijector only) in {time.time()-t0:.0f}s",
          flush=True)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    out = dict(labels=np.array(OLD_LABELS_46))
    prov = dict(script="25a_export_v3b_basin_x46.py",
                generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                convention="prod_smc fit['<basin>_draws'] -> bij.forward -> "
                           "_flat_named46, canonical OLD_LABELS_46 order",
                max_draws=MAX_DRAWS, subsample_seed=SUBSAMPLE_SEED, inputs={})
    for basin, path in (("low", args.low_npz), ("steep", args.steep_npz)):
        p = Path(path)
        fit = np.load(p, allow_pickle=True)
        key = f"{basin}_draws"
        assert key in fit, f"{p} lacks {key}; keys={sorted(fit.keys())}"
        draws = np.asarray(fit[key], dtype=np.float64)
        if draws.ndim == 3:
            draws = draws.reshape(-1, draws.shape[-1])
        n_total = draws.shape[0]
        if n_total > MAX_DRAWS:
            draws = draws[rng.choice(n_total, MAX_DRAWS, replace=False)]
        labeled = dict(_flat_named46_batched(
            model.bij.forward(list(jnp.asarray(draws).T))))
        x46 = np.stack([labeled[lab] for lab in OLD_LABELS_46], axis=1)
        assert x46.shape == (draws.shape[0], 46)
        assert np.all(np.isfinite(x46)), f"{basin}: non-finite constrained draws"
        gam = x46[:, OLD_LABELS_46.index("mass.gamma")]
        out[f"{basin}_x46"] = x46
        prov["inputs"][basin] = dict(file=str(p), md5=md5(p), key=key,
                                     n_total=int(n_total), n_kept=int(x46.shape[0]),
                                     gamma_med=float(np.median(gam)))
        print(f"[25a] {basin}: {x46.shape[0]} draws, gamma_med "
              f"{np.median(gam):.4f} (source {p.name} md5 {prov['inputs'][basin]['md5'][:8]})",
              flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out["provenance"] = json.dumps(prov)
    np.savez(outp, **out)
    print(f"[25a] wrote {outp} (md5 {md5(outp)})", flush=True)


if __name__ == "__main__":
    main()
