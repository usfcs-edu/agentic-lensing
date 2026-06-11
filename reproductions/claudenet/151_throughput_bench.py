#!/usr/bin/env python3
"""151_throughput_bench.py — Phase 150: measured cutouts/s/GPU for the distilled
student, every v1/v2 member checkpoint, and the full shared-load ensemble
stage-1 — plus the Phase-150 gate check (runs LOCALLY, one GPU; pin with
CUDA_VISIBLE_DEVICES from the caller).

BENCH METHOD (no v1 bench script survives; the v1 paper table — shielded-194k
2,843, EfficientNetV2-S 5,594, 5-member shared-load 865 c/s — states "TITAN
RTX, batch 256, compute-only upper bounds", so we mirror that): batch 256 of
random pre-normalised (B,3,101,101) tensors resident on the device, 30 warmup
+ 100 timed batches through _trainlib.model_prob, torch.cuda.synchronize()
around each timing; report mean +/- std of per-batch cutouts/s. The shared-load
ensemble row runs EVERY stage-1 member sequentially on the SAME batch (the
scan pays the FITS load once). TF32 is forced off (112/100 parity).
SYNTHETIC-TENSOR CAVEAT: these are compute-only upper bounds — FITS I/O,
normalisation and host->device copies are excluded, exactly as in the v1 table;
real scans are I/O-bound.

PHASE-150 GATE (plan: "student >= 4,300 c/s AND student recovery@0.1%FPR >=
ensemble - 0.02 on both negative sets"):
  * throughput: measured student c/s >= --min-cs (4300).
  * accuracy, negative set #1 (v1 testneg): student threshold from its OWN
    testneg scores (data/v2/scores_student_distilled.parquet), recovery of the
    storfer/inchausti positives (v1-verbatim _ensemble arithmetic); ensemble
    reference = the flagship's oldset_rec_* columns of the 145/113 operating-
    points CSV (data/v2/ensemble_v2_operating_points.csv preferred — 145
    reuses 113's rows_from_results schema, flagship scorer 'v2:average' —
    falling back to 113's data/v2/operating_points_v2.csv, i.e. the v1-roster
    'average' numbers, labelled so).
  * accuracy, negative set #2 (NegEval-1M): ensemble reference = the same
    CSV's rec_* columns; the student needs pool scores — a
    member_student_distilled / student_distilled column in --pool-scores, or a
    separate --student-pool-scores parquet (produce it on Perlmutter with
    `112_score_pool.py --extra-ckpt-dir data/v2/ckpt_student --only-extra`).
  Matched-FPR thresholds/recovery are invariant under the monotone isotonic
  calibration, so raw student p is compared honestly to the calibrated
  ensemble. Any missing input -> gate INCOMPLETE (never silently passed).
  Exit code: 0 PASS / 1 FAIL / 2 INCOMPLETE.

Writes data/v2/throughput_v2.json (bench + gate; --gate-only updates the gate
block in place without touching a GPU) + a printed table.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
      /home2/benson/.venvs/claudenet/bin/python 151_throughput_bench.py
    # accuracy-gate refresh only (CPU, no bench):
    python 151_throughput_bench.py --gate-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C
import _ensemble as E
import _trainlib as TL

V2 = C.DATA / "v2"
V1_REFERENCE = {"shielded_194k": 2843, "resnet46": 3349, "effnetv2_s": 5594,
                "effnet_b3": 6728, "five_member_shared_load": 865}
STAGE1_V1 = "shielded_A,effnet_B,effnet_B3,effnet_S2,resnet46_C"
STUDENT_COLS = ("member_student_distilled", "student_distilled")


# ===== bench core ==============================================================

def bench(fn, batch: int, warmup: int, timed: int, device) -> tuple[float, float]:
    """Time `fn` on one resident random batch; return (mean, std) cutouts/s
    over the timed batches (compute-only: no I/O, no normalise, no H2D copy)."""
    x = torch.randn(batch, 3, 101, 101, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(warmup + timed):
        t0 = time.perf_counter()
        fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    cs = batch / np.asarray(times[warmup:], dtype=np.float64)
    return float(cs.mean()), float(cs.std())


def collect_checkpoints(args) -> dict[str, Path]:
    """{model_key: ckpt path} — student + v1 members + v2 members (smoke ckpts
    skipped)."""
    out = {}
    student = Path(args.student_ckpt)
    if student.exists():
        out["student_distilled"] = student
    else:
        print(f"[bench] NOTE: student checkpoint {student} missing -> not benched")
    for tag, d in (("v1", Path(args.v1_ckpt_dir)), ("v2", Path(args.v2_ckpt_dir))):
        for p in sorted(d.glob("member_*.pt")) if d.exists() else []:
            if p.stem.endswith("_smoke"):
                continue
            key = p.stem if tag == "v1" else f"{p.stem}_v2"
            out[key] = p
    return out


def run_bench(args, device) -> dict:
    M112 = C._load("cn_151_m112", C.ROOT / "112_score_pool.py")
    ckpts = collect_checkpoints(args)
    models, results = {}, {}
    for key, path in ckpts.items():
        name = path.stem[len("member_"):] if path.stem.startswith("member_") else path.stem
        try:
            model, score_arch, _, _ = M112.load_member_checkpoint(
                path, device, M112.member_variant(name, path.parent.parent))
        except Exception as e:
            print(f"[bench] SKIP {key}: cannot load ({e})")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        fn = lambda x, m=model, a=score_arch: TL.model_prob(m, x, a)
        mu, sd = bench(fn, args.batch, args.warmup, args.timed, device)
        results[key] = {"ckpt": str(path), "params": int(n_params),
                        "score_arch": score_arch, "cutouts_per_s_mean": mu,
                        "cutouts_per_s_std": sd}
        models[key] = (model, score_arch)
        print(f"[bench] {key:32s} params={n_params:>12,}  "
              f"{mu:>8.0f} +/- {sd:>6.0f} c/s")

    # shared-load stage-1: all CNN members, one batch load. v2 checkpoints are
    # benched under a '_v2'-suffixed key — accept bare AND suffixed names.
    def resolve(s: str) -> str:
        base = s if s.startswith("member_") else f"member_{s}"
        return next((c for c in (base, f"{base}_v2") if c in models), base)

    want = [] if args.stage1 == "none" else (
        [k for k in models if k != "student_distilled"] if args.stage1 == "all"
        else [resolve(s) for s in args.stage1.split(",") if s.strip()])
    stage1 = [k for k in want if k in models]
    missing = sorted(set(want) - set(stage1))
    ens = None
    if missing:
        print(f"[bench] stage-1 members missing from the bench: {missing}")
    if stage1:
        def fn_ens(x, ms=[models[k] for k in stage1]):
            for m, a in ms:
                TL.model_prob(m, x, a)
        mu, sd = bench(fn_ens, args.batch, args.warmup, args.timed, device)
        ens = {"members": stage1, "n_members": len(stage1),
               "cutouts_per_s_mean": mu, "cutouts_per_s_std": sd}
        print(f"[bench] {'ensemble_stage1 (' + str(len(stage1)) + ' shared-load)':32s} "
              f"{'':>21s}{mu:>8.0f} +/- {sd:>6.0f} c/s")
    return {"models": results, "ensemble_stage1": ens, "stage1_missing": missing}


# ===== Phase-150 gate ==========================================================

def _student_pool_scores(args, missing: list[str]) -> np.ndarray | None:
    """Student raw scores on the NegEval-1M pool, from --student-pool-scores or
    a student column merged into --pool-scores."""
    f = Path(args.student_pool_scores) if args.student_pool_scores else None
    if f is None or not f.exists():
        f = Path(args.pool_scores)
    if not f.exists():
        missing.append(f"pool scores parquet ({f})")
        return None
    import pyarrow.parquet as pq                 # cheap column probe, no full read
    have = set(pq.ParquetFile(f).schema_arrow.names)
    col = next((c for c in STUDENT_COLS if c in have), None)
    if col is None:
        missing.append(f"student column {STUDENT_COLS} in {f} (run 112 "
                       f"--extra-ckpt-dir data/v2/ckpt_student --only-extra on Perlmutter)")
        return None
    v = pd.read_parquet(f, columns=[col])[col].to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    print(f"[gate] student pool scores: {f.name}:{col} ({len(v):,} finite rows)")
    return v


def _ensemble_reference(args, missing: list[str]):
    """Flagship recovery@--fpr from the 145 (preferred) / 113 operating-points
    CSV; --ensemble-scorer is a comma list of scorer names tried in order
    (145 prefixes the roster tag: 'v2:average'; 113 uses bare 'average').
    Returns (source_name, {('testneg'|'negeval_pool', 'storfer'|'inchausti'): rec})."""
    cands = ([Path(args.operating_points)] if args.operating_points else
             [V2 / "ensemble_v2_operating_points.csv", V2 / "operating_points_v2.csv"])
    f = next((p for p in cands if p.exists()), None)
    if f is None:
        missing.append(f"ensemble operating-points CSV (none of "
                       f"{[str(p) for p in cands]}; run 145/113 first)")
        return None, {}
    if not args.operating_points and f != cands[0]:
        print(f"[gate] WARNING: {cands[0]} missing -> FALLING BACK to 113's {f} "
              f"(v1-roster reference, NOT the 145 v2 flagship — run 145 for the "
              f"real gate); verdict json tags it as 'recovery_reference'")
    df = pd.read_csv(f)
    scorers = [s.strip() for s in args.ensemble_scorer.split(",") if s.strip()]
    sub, scorer = None, None
    for s in scorers:
        m = df[(df.scorer == s) & np.isclose(df.fpr, args.fpr)]
        if len(m) == 1:
            sub, scorer = m, s
            break
    if sub is None:
        missing.append(f"{f.name}: no unique row for scorer in {scorers} at "
                       f"fpr={args.fpr:g} (have: {sorted(df.scorer.unique())})")
        return str(f), {}
    r = sub.iloc[0]
    ref = {}
    for sp in ("storfer", "inchausti"):
        if np.isfinite(r.get(f"rec_{sp}", np.nan)):
            ref[("negeval_pool", sp)] = float(r[f"rec_{sp}"])
        else:
            missing.append(f"{f.name}: rec_{sp} for {scorer}")
        if np.isfinite(r.get(f"oldset_rec_{sp}", np.nan)):
            ref[("testneg", sp)] = float(r[f"oldset_rec_{sp}"])
        else:
            missing.append(f"{f.name}: oldset_rec_{sp} for {scorer}")
    tag = " (v1-roster numbers from 113)" if f.name == "operating_points_v2.csv" else ""
    print(f"[gate] ensemble reference: {f.name} scorer={scorer}{tag}")
    return str(f), ref


def gate_check(args, student_cs: float | None) -> dict:
    missing: list[str] = []
    gate = {"fpr": args.fpr, "tolerance": args.tolerance,
            "min_cutouts_per_s": args.min_cs, "ensemble_scorer": args.ensemble_scorer,
            "student_cutouts_per_s": student_cs}

    if student_cs is None:
        missing.append("student throughput (bench not run / student ckpt missing)")
        gate["throughput_pass"] = None
    else:
        gate["throughput_pass"] = bool(student_cs >= args.min_cs)

    src, ref = _ensemble_reference(args, missing)
    gate["ensemble_source"] = src
    gate["recovery_reference"] = src          # which operating-points CSV was used

    # student scores on the v1 eval splits
    sf = Path(args.student_scores)
    stu = {}
    if sf.exists():
        s = pd.read_parquet(sf)
        neg_testneg = s[s.split == "testneg"]["p"].to_numpy(dtype=np.float64)
        for sp in ("storfer", "inchausti"):
            stu[("pos", sp)] = s[s.split == sp]["p"].to_numpy(dtype=np.float64)
        thr = E.fpr_threshold(neg_testneg, args.fpr)
        for sp in ("storfer", "inchausti"):
            stu[("testneg", sp)] = float((stu[("pos", sp)] >= thr).mean())
    else:
        missing.append(f"student eval scores ({sf}; run 150 first)")

    pool = _student_pool_scores(args, missing)
    if pool is not None and sf.exists():
        thr = E.fpr_threshold(pool, args.fpr)
        for sp in ("storfer", "inchausti"):
            stu[("negeval_pool", sp)] = float((stu[("pos", sp)] >= thr).mean())

    rec, ok_all = {}, True
    for negset in ("testneg", "negeval_pool"):
        rec[negset] = {}
        for sp in ("storfer", "inchausti"):
            s_rec = stu.get((negset, sp))
            e_rec = ref.get((negset, sp))
            ent = {"student": s_rec, "ensemble": e_rec, "delta": None, "pass": None}
            if s_rec is not None and e_rec is not None:
                ent["delta"] = s_rec - e_rec
                ent["pass"] = bool(ent["delta"] >= -args.tolerance)
                ok_all &= ent["pass"]
            elif s_rec is None and sf.exists() and negset == "negeval_pool":
                pass  # pool-side miss already recorded
            rec[negset][sp] = ent
    gate["recovery"] = rec
    gate["missing"] = missing
    n_done = sum(1 for ns in rec.values() for e in ns.values() if e["pass"] is not None)
    if missing or n_done < 4 or gate["throughput_pass"] is None:
        gate["verdict"] = "INCOMPLETE"
    else:
        gate["verdict"] = "PASS" if (gate["throughput_pass"] and ok_all) else "FAIL"

    print(f"\n[gate] Phase-150 gate @ FPR {args.fpr:g} (tol {args.tolerance}, "
          f"student >= {args.min_cs} c/s):")
    print(f"[gate]   throughput: student "
          f"{('%.0f' % student_cs) if student_cs is not None else 'MISSING'} c/s "
          f"-> {gate['throughput_pass']}")
    for negset, ns in rec.items():
        for sp, e in ns.items():
            fmt = lambda v: "MISSING" if v is None else f"{v:.3f}"
            print(f"[gate]   {negset:12s}|{sp:9s} student={fmt(e['student'])} "
                  f"ensemble={fmt(e['ensemble'])} "
                  f"delta={'MISSING' if e['delta'] is None else format(e['delta'], '+.3f')} "
                  f"-> {e['pass']}")
    for m in missing:
        print(f"[gate]   MISSING: {m}")
    print(f"[gate] VERDICT: {gate['verdict']}")
    return gate


# ===== main ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu (tiny smokes)")
    ap.add_argument("--student-ckpt", default=str(V2 / "ckpt" / "student_distilled.pt"))
    ap.add_argument("--v1-ckpt-dir", default=str(C.CKPT), help="v1 member_*.pt dir")
    ap.add_argument("--v2-ckpt-dir", default=str(V2 / "ckpt"), help="v2 member_*.pt dir")
    ap.add_argument("--stage1", default=STAGE1_V1,
                    help="shared-load stage-1 roster: comma member names (default = "
                         "the 5 v1 members, mirroring the v1 865 c/s row), 'all' "
                         "(every benched member), or 'none'; v2 checkpoints match "
                         "with or without their '_v2' bench-key suffix")
    ap.add_argument("--out", default=str(V2 / "throughput_v2.json"))
    ap.add_argument("--gate-only", action="store_true",
                    help="skip the GPU bench; refresh the gate block from the "
                         "score/CSV artifacts (CPU only)")
    # gate inputs
    ap.add_argument("--student-scores", default=str(V2 / "scores_student_distilled.parquet"))
    ap.add_argument("--pool-scores", default=str(V2 / "scores_negeval_pool.parquet"))
    ap.add_argument("--student-pool-scores", default=None,
                    help="separate parquet with a member_student_distilled column "
                         "(from 112 --extra-ckpt-dir data/v2/ckpt_student)")
    ap.add_argument("--operating-points", default=None,
                    help="ensemble operating-points CSV (default: try 145's "
                         "ensemble_v2_operating_points.csv then 113's "
                         "operating_points_v2.csv)")
    ap.add_argument("--ensemble-scorer", default="v2:average,average",
                    help="comma list of flagship scorer rows tried in order "
                         "(145 tags rosters: 'v2:average'; 113 uses 'average')")
    ap.add_argument("--fpr", type=float, default=1e-3)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--min-cs", type=float, default=4300.0)
    args = ap.parse_args()

    out_f = Path(args.out)
    out_f.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out_f.read_text()) if out_f.exists() else {}

    if args.gate_only:
        prev = (payload.get("models") or {}).get("student_distilled") or {}
        student_cs = prev.get("cutouts_per_s_mean")
        print(f"[151] --gate-only (no GPU); student c/s from {out_f.name}: {student_cs}")
    else:
        # TF32 off for parity with the scoring harness (cf. 112/100)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[151] FATAL: --device cuda but no GPU visible")
            return 1
        dev_name = (torch.cuda.get_device_name(device) if device.type == "cuda"
                    else "cpu")
        print(f"[151] device={dev_name} batch={args.batch} warmup={args.warmup} "
              f"timed={args.timed} tf32=off (compute-only synthetic tensors — "
              f"excludes FITS I/O, as the v1 table)")
        b = run_bench(args, device)
        payload.update({"device": dev_name, "batch": args.batch,
                        "warmup": args.warmup, "timed": args.timed,
                        "v1_reference_cutouts_per_s": V1_REFERENCE, **b})
        stu = payload["models"].get("student_distilled") or {}
        student_cs = stu.get("cutouts_per_s_mean")

    payload["gate"] = gate_check(args, student_cs)
    out_f.write_text(json.dumps(payload, indent=2))
    print(f"[151] wrote {out_f}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[payload["gate"]["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
