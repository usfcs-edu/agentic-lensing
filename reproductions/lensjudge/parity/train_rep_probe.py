#!/usr/bin/env python3
"""Phase C1b: the checkpoint-free conventional-ML baseline for the parity bench.

A calibrated logistic probe on the engineered Tier-1 lensing features
(common/representations.compute_features), trained ONLY on the leak-aware pool
(outputs/parity_train_pool.csv, built by parity/build_train_splits.py):

  train subset:  every graded A/B/C + every graded_D row of split=train, plus a
                 seed-2026 sample of N_RANDOM_TRAIN random-galaxy negatives.
  valsel:        every graded/graded_D row + N_RANDOM_VALSEL seed-2026 randoms —
                 hyperparameter selection (C, label scheme) by AUC on the HARD
                 contrast graded A/B vs graded_D, then isotonic calibration of
                 probe score -> p_lens against the pool's human-soft targets.
  gate:          FROZEN. Evaluated exactly once (--gate; refuses to rerun if
                 outputs/rep_probe/gate_report.json already exists). Nothing is
                 ever fitted or selected on gate.

The deployed score is p_lens = 0.995*isotonic(s) + 0.005*s (s = raw logistic
probability): the 0.005 blend keeps the ranking strictly monotone in s (isotonic
alone introduces ties) while distorting calibration by <= 0.005.

Feature vectors are cached in outputs/rep_probe/features_cache.parquet (keyed by
candidate name; label-free) and shared with score_bench_rep.py. Cutouts resolve
via common/fetch.get_cube: on-disk/cache cubes are computed with LOCAL_WORKERS
threads; endpoint misses are fetched politely with FETCH_WORKERS (<=3) threads.

  python lensjudge/parity/train_rep_probe.py --smoke   # tiny run, never touches gate
  python lensjudge/parity/train_rep_probe.py --gate    # full train + one-shot gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch, representations as R  # noqa: E402

SEED = 2026
OUT = config.OUT / "rep_probe"
POOL_CSV = config.OUT / "parity_train_pool.csv"
CACHE_PARQUET = OUT / "features_cache.parquet"
MODEL_PATH = OUT / "rep_probe.joblib"
GATE_REPORT = OUT / "gate_report.json"

N_RANDOM_TRAIN = 3000
N_RANDOM_VALSEL = 600
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LABEL_SCHEMES = ("binary", "soft")   # binary: graded=1 vs D/random=0; soft: v5 soft targets
N_BOOT = 2000
FETCH_WORKERS = 3                    # polite ceiling against legacysurvey.org
LOCAL_WORKERS = 12
CHECKPOINT_EVERY = 300


# ------------------------------------------------------------------ feature cache
def _is_local(name: str, survey: str) -> bool:
    if fetch.on_disk_path(name, survey if survey in config.CUTOUT_DIRS else None):
        return True
    p = config.CACHE / "cubes" / f"{name}.fits"
    return p.exists() and p.stat().st_size > 256


def load_feature_cache() -> pd.DataFrame:
    if CACHE_PARQUET.exists():
        return pd.read_parquet(CACHE_PARQUET)
    return pd.DataFrame(columns=["name"])


def _write_cache(cache: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PARQUET.with_suffix(".tmp")
    cache.to_parquet(tmp, index=False)
    tmp.rename(CACHE_PARQUET)


def compute_features_for(rows: pd.DataFrame, fetch_workers: int = FETCH_WORKERS,
                         local_workers: int = LOCAL_WORKERS) -> pd.DataFrame:
    """Ensure the feature cache covers every row (name, ra, dec, survey_key).

    Returns the full cache. Rows whose cube cannot be resolved are absent from it.
    """
    cache = load_feature_cache()
    have = set(cache["name"].astype(str)) if len(cache) else set()
    todo = rows.drop_duplicates("name")
    todo = todo[~todo["name"].astype(str).isin(have)]
    if not len(todo):
        return cache
    local_mask = todo.apply(lambda r: _is_local(str(r["name"]), str(r["survey_key"])), axis=1)
    batches = [("local", todo[local_mask], local_workers),
               ("remote", todo[~local_mask], fetch_workers)]

    lock = threading.Lock()
    new_rows: list[dict] = []
    stats = {"ok": 0, "fail": 0}

    def one(r) -> dict | None:
        try:
            cube = fetch.get_cube(name=str(r["name"]), ra=r.get("ra"), dec=r.get("dec"),
                                  survey=str(r.get("survey_key", "storfer")))
        except Exception:
            cube = None
        if cube is None:
            return None
        return {"name": str(r["name"]), **R.compute_features(cube)}

    for tag, batch, workers in batches:
        if not len(batch):
            continue
        print(f"[features] {tag}: {len(batch)} cubes ({workers} workers) ...", flush=True)
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, r) for _, r in batch.iterrows()]
            for fut in as_completed(futs):
                res = fut.result()
                with lock:
                    done += 1
                    if res is not None:
                        new_rows.append(res)
                        stats["ok"] += 1
                    else:
                        stats["fail"] += 1
                    if done % 100 == 0 or done == len(batch):
                        dt = time.time() - t0
                        eta = dt / done * (len(batch) - done)
                        print(f"[features] {tag} {done}/{len(batch)} "
                              f"(ok={stats['ok']} fail={stats['fail']}) "
                              f"{dt:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
                    if len(new_rows) and len(new_rows) % CHECKPOINT_EVERY == 0:
                        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                        cache = cache.drop_duplicates("name")
                        _write_cache(cache)
                        new_rows = []
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates("name")
    _write_cache(cache)
    print(f"[features] cache now {len(cache)} rows (+{stats['ok']} new, {stats['fail']} unresolved)",
          flush=True)
    return cache


def feature_cols(cache: pd.DataFrame) -> list[str]:
    return [c for c in cache.columns if c != "name"]


# ------------------------------------------------------------------ pool subsets
def load_pool() -> pd.DataFrame:
    pool = pd.read_csv(POOL_CSV, dtype={"grade": str})
    pool["grade"] = pool["grade"].fillna("")
    sha_file = Path(str(POOL_CSV) + ".sha")
    if sha_file.exists():
        h = hashlib.sha256(POOL_CSV.read_bytes()).hexdigest()[:16]
        pinned = sha_file.read_text().strip()
        if h != pinned:
            print(f"[pool] WARNING sha mismatch: file {h} vs pinned {pinned}", flush=True)
        else:
            print(f"[pool] sha OK ({h})", flush=True)
    return pool


def _sample_random(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministic seed-2026 sample of random_neg rows, order-independent."""
    rand = sub[sub.label_source == "random_neg"].sort_values("name").reset_index(drop=True)
    if len(rand) <= n:
        return rand
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(rand), size=n, replace=False))
    return rand.iloc[idx]


def select_subsets(pool: pd.DataFrame, smoke: bool = False):
    def block(split: str, n_random: int) -> pd.DataFrame:
        sub = pool[pool.split == split]
        graded = sub[sub.label_source.isin(["graded", "graded_D"])]
        rand = _sample_random(sub, n_random)
        return pd.concat([graded, rand], ignore_index=True)

    train = block("train", N_RANDOM_TRAIN)
    valsel = block("valsel", N_RANDOM_VALSEL)
    gate = pool[pool.split == "gate"].reset_index(drop=True)
    if smoke:
        rng = np.random.default_rng(SEED)
        def shrink(df: pd.DataFrame, per_source: int) -> pd.DataFrame:
            keep = []
            for _, grp in df.groupby("label_source"):
                grp = grp.sort_values("name").reset_index(drop=True)
                idx = np.sort(rng.choice(len(grp), size=min(per_source, len(grp)), replace=False))
                keep.append(grp.iloc[idx])
            return pd.concat(keep, ignore_index=True)
        train, valsel = shrink(train, 60), shrink(valsel, 25)
        gate = valsel.copy()   # SMOKE pseudo-gate: the real gate is never touched
    return train, valsel, gate


# ------------------------------------------------------------------ fit + select
def _xy(df: pd.DataFrame, cache: pd.DataFrame, cols: list[str]):
    m = df.merge(cache, on="name", how="inner")
    X = m[cols].fillna(0).values.astype(float)
    return m, X


def fit_probe(train_m: pd.DataFrame, X: np.ndarray, C: float, scheme: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))  # default penalty: L2
    if scheme == "binary":
        y = (train_m.label_source == "graded").astype(int).values
        pipe.fit(X, y)
    else:                                   # soft: v5 human-soft p_lens targets
        soft = train_m.soft_target.values.astype(float)
        X2 = np.vstack([X, X])
        y2 = np.r_[np.ones(len(X)), np.zeros(len(X))]
        w2 = np.r_[soft, 1.0 - soft]
        pipe.fit(X2, y2, logisticregression__sample_weight=w2)
    return pipe


def hard_masks(m: pd.DataFrame):
    pos = (m.label_source == "graded") & m.grade.isin(["A", "B"])
    neg = m.label_source == "graded_D"
    return pos.values, neg.values


def auc_ci(pos_scores, neg_scores, n_boot: int = N_BOOT, seed: int = SEED):
    """Directional AUC (probe high = lens) with a stratified bootstrap CI."""
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(pos_scores, float); neg = np.asarray(neg_scores, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), (float("nan"), float("nan")), len(pos), len(neg)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    a = float(roc_auc_score(y, np.r_[pos, neg]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        ip = rng.integers(0, len(pos), len(pos))
        ineg = rng.integers(0, len(neg), len(neg))
        try:
            boot.append(roc_auc_score(y, np.r_[pos[ip], neg[ineg]]))
        except ValueError:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return a, (float(lo), float(hi)), len(pos), len(neg)


def calibrated_p(pipe, iso, X: np.ndarray) -> np.ndarray:
    s = pipe.predict_proba(X)[:, 1]
    return 0.995 * iso.transform(s) + 0.005 * s


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subsets, pseudo-gate=valsel copy; saves no model/report")
    ap.add_argument("--gate", action="store_true",
                    help="run the frozen gate evaluation (exactly once)")
    ap.add_argument("--fetch-workers", type=int, default=FETCH_WORKERS)
    ap.add_argument("--local-workers", type=int, default=LOCAL_WORKERS)
    args = ap.parse_args()

    if args.gate and GATE_REPORT.exists():
        sys.exit(f"REFUSING: {GATE_REPORT} exists — the gate is evaluated exactly once.")
    if args.gate and args.smoke:
        sys.exit("REFUSING: --gate with --smoke makes no sense.")

    t_start = time.time()
    pool = load_pool()
    train, valsel, gate = select_subsets(pool, smoke=args.smoke)
    print(f"[subsets] train={len(train)} valsel={len(valsel)} gate={len(gate)}"
          f"{' (SMOKE pseudo-gate)' if args.smoke else ''}", flush=True)

    need = pd.concat([train, valsel] + ([gate] if (args.gate or args.smoke) else []),
                     ignore_index=True)[["name", "ra", "dec", "survey_key"]]
    cache = compute_features_for(need, args.fetch_workers, args.local_workers)
    cols = feature_cols(cache)

    train_m, Xtr = _xy(train, cache, cols)
    val_m, Xval = _xy(valsel, cache, cols)
    print(f"[coverage] train {len(train_m)}/{len(train)}  valsel {len(val_m)}/{len(valsel)}",
          flush=True)

    # -- hyperparameter selection on valsel (HARD: graded A/B vs graded_D) --------
    vpos, vneg = hard_masks(val_m)
    results = []
    for scheme in LABEL_SCHEMES:
        for C in C_GRID:
            pipe = fit_probe(train_m, Xtr, C, scheme)
            s = pipe.predict_proba(Xval)[:, 1]
            a, _, _, _ = auc_ci(s[vpos], s[vneg], n_boot=0)
            results.append((scheme, C, a))
            print(f"[valsel] scheme={scheme:6s} C={C:<5g} HARD-AUC={a:.4f}", flush=True)
    scheme, C, best_auc = max(results, key=lambda t: (t[2], -t[1]))
    print(f"[valsel] SELECTED scheme={scheme} C={C} (HARD AUC {best_auc:.4f})", flush=True)

    pipe = fit_probe(train_m, Xtr, C, scheme)

    # -- isotonic calibration on valsel against the pool's human-soft targets -----
    from sklearn.isotonic import IsotonicRegression
    s_val = pipe.predict_proba(Xval)[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s_val, val_m.soft_target.values.astype(float))

    # -- feature importance (coefficients on standardized features) ---------------
    lr = pipe.named_steps["logisticregression"]
    coef = pd.Series(lr.coef_[0], index=cols).sort_values(key=np.abs, ascending=False)
    print("[probe] top-8 |coef| (standardized features):", flush=True)
    for k, v in coef.head(8).items():
        print(f"   {k:30s} {v:+.3f}")

    if not args.smoke:
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipe, "isotonic": iso, "feature_cols": cols,
                     "C": C, "scheme": scheme, "valsel_hard_auc": best_auc,
                     "seed": SEED, "n_train": len(train_m), "n_valsel": len(val_m),
                     "blend": "p_lens = 0.995*iso(s) + 0.005*s"},
                    MODEL_PATH)
        print(f"[probe] model -> {MODEL_PATH}", flush=True)

    # -- gate: exactly once --------------------------------------------------------
    if args.gate or args.smoke:
        tag = "SMOKE-pseudo-gate" if args.smoke else "GATE"
        gate_m, Xg = _xy(gate, cache, cols)
        print(f"[{tag}] coverage {len(gate_m)}/{len(gate)}", flush=True)
        p = calibrated_p(pipe, iso, Xg)
        graded = gate_m.label_source == "graded"
        contrasts = {
            "hard_AB_vs_gradedD": (graded & gate_m.grade.isin(["A", "B"]),
                                   gate_m.label_source == "graded_D"),
            "easy_A_vs_random": (graded & (gate_m.grade == "A"),
                                 gate_m.label_source == "random_neg"),
            "ABC_vs_random": (graded & gate_m.grade.isin(["A", "B", "C"]),
                              gate_m.label_source == "random_neg"),
        }
        report = {"hyperparams": {"C": C, "scheme": scheme}, "seed": SEED,
                  "n_boot": N_BOOT, "score": "p_lens_rep (isotonic-calibrated, 0.005 blend)",
                  "valsel_hard_auc": round(best_auc, 4),
                  "gate_coverage": f"{len(gate_m)}/{len(gate)}",
                  "feature_top5": {k: round(float(v), 4) for k, v in coef.head(5).items()},
                  "contrasts": {}}
        for cname, (pm, nm) in contrasts.items():
            a, (lo, hi), npos, nneg = auc_ci(p[pm.values], p[nm.values])
            report["contrasts"][cname] = {"auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                          "n_pos": npos, "n_neg": nneg}
            print(f"[{tag}] {cname:22s} AUC={a:.3f} [{lo:.3f},{hi:.3f}] "
                  f"({npos} pos vs {nneg} neg)", flush=True)
        if not args.smoke:
            GATE_REPORT.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[GATE] report -> {GATE_REPORT}", flush=True)

    print(f"[done] {time.time() - t_start:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
