#!/usr/bin/env python3
"""161_sweep_score.py — Phase 160: stage-1 scoring wrapper for the full-DR9
sweep — one shared-GPU slurm job per (part root, shard range) (runs ON
PERLMUTTER; --plan also runs anywhere that can read the extraction roots).

This is a thin, collision-safe wrapper around 112_score_pool.py's machinery
(subprocess, the 112->112b pattern): it derives a UNIQUE --out per
(part, shard range, stage-1 mode) inside --out-dir, so 112's resume sidecars
(<out>.partial_<col>.npz, named after --out and fingerprinted to the filtered
index) can never collide between concurrently running ranges, then invokes

  --stage1 members  ->  112 --skip-baselines --shard-range A:B
                        (the five v1 CNN members; baselines are
                        wasted GPU-min at 17.3M scale and play no part in
                        stage-1 selection)
  --stage1 lean     ->  112 --only-extra --extra-ckpt-dir <--lean-ckpt-dir>
                        --shard-range A:B   (the LEAN v2 roster
                        data/v2/roster_v2lean.json: effnet_B + effnet_B3_hard +
                        effnet_S2_hard + resnet46_C_hard + zoobot_N; columns =
                        ckpt stems == the persisted v2lean fits' pool_columns.
                        <--lean-ckpt-dir> must hold EXACTLY those five
                        member_<name>.pt files — asserted before the GPU pass)
  --stage1 student  ->  112 --only-extra --extra-ckpt-dir <--student-ckpt-dir>
                        --shard-range A:B   (the 150 distilled student,
                        column member_student_distilled)

The 150 gate decides which mode the orchestrator deploys; both are first-class
here. STUDENT-MODE PREREQUISITES: besides the 150 checkpoint
(<--student-ckpt-dir>/member_student_distilled.pt), 162 can only threshold the
student once a 113-style run over the student's NegEval scores has appended a
'student_distilled' row to operating_points_v2.csv (+ its 'evt' block and an
isotonic calibrator in the persisted 145 fits) — without those, 162
--stage1-mode student FATALs at the threshold lookup.
Resume-safe at two levels: an existing final --out parquet (112 writes it
atomically) skips the 112 call entirely; a killed job resumes AT THE LAST
COMPLETED CHECKPOINT PASS — 112 writes its fingerprinted partial npz only
after a full pass of one checkpoint over the shard range, so a mid-pass kill
redoes that pass (up to 1 of 5 member passes; the single-pass student restarts
its range). Each finished range writes a sidecar json (<out>.json:
n_rows/n_ok/columns/elapsed) for OPERATOR/AUDIT-TRAIL use only —
162 --make-survivors audits coverage directly against the 160 part manifests
(the stronger row_id-level check) and does not read these sidecars.

SBATCH FAN-OUT (the orchestrator submits; --plan prints these lines): the
sweep extraction (160 -> K x 111) leaves K part roots, each with shards
cutouts_0..N-1.npy of ~--shard-size (50k) rows. One full 5-CNN pass over 1M
rows took ~43 GPU-min incl. baselines, so ~8 shards (400k rows) per shared-GPU
job is a comfortable <1 h unit; the student is ~6x faster -> 32+ shards/job.

    # enumerate the jobs for one part root (no GPU needed):
    python 161_sweep_score.py --plan --shards-per-job 8 \
        --cutout-root $SCRATCH/claudenet/cutouts/sweep/part00 \
        --out-dir $SCRATCH/claudenet/scores/sweep --stage1 members
    # one emitted line looks like:
    sbatch --export=ALL,CMD='python 161_sweep_score.py \
        --cutout-root $SCRATCH/claudenet/cutouts/sweep/part00 \
        --shard-range 0:8 --stage1 members \
        --out-dir $SCRATCH/claudenet/scores/sweep' nersc/shared_gpu.slurm
    # afterwards: rsync $SCRATCH/claudenet/scores/sweep/*.parquet (+ .json)
    # back to data/v2/sweep/stage1/ and run 162 --make-survivors locally.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

STAGE1_COLS = {"members": [f"member_{n}" for n in
                           ("shielded_A", "effnet_B", "effnet_B3",
                            "effnet_S2", "resnet46_C")],
               # LEAN v2 roster (data/v2/roster_v2lean.json) — MUST equal the
               # persisted v2lean fits' pool_columns AND the member_<name>.pt
               # stems in --lean-ckpt-dir (112 --only-extra names columns by stem)
               "lean": [f"member_{n}" for n in
                        ("effnet_B", "effnet_B3_hard", "effnet_S2_hard",
                         "resnet46_C_hard", "zoobot_N")],
               "student": None}     # student: columns = ckpt stems (discovered)


def parse_range(spec: str) -> tuple[int, int]:
    try:
        a, b = (int(x) for x in spec.split(":"))
    except ValueError:
        raise SystemExit(f"[161] FATAL: --shard-range {spec!r}: expected A:B integers")
    if not 0 <= a < b:
        raise SystemExit(f"[161] FATAL: --shard-range {spec!r}: need 0 <= A < B")
    return a, b


def shard_ids(root: Path) -> list[int]:
    """Shard ids present in the part root's 111 index (authoritative)."""
    idx = pd.read_parquet(root / "index.parquet", columns=["shard"])
    return sorted(int(s) for s in idx["shard"].unique())


def out_path(out_dir: Path, part_tag: str, mode: str, a: int, b: int) -> Path:
    """The collision-safety contract: one file name per (part, mode, range),
    so 112's <out>.partial_<col>.npz sidecars are disjoint across jobs."""
    return out_dir / f"stage1_{part_tag}_{mode}_s{a:05d}-{b:05d}.parquet"


def plan(args) -> int:
    """Print the sbatch fan-out for one part root (orchestrator submits)."""
    root = Path(args.cutout_root)
    ids = shard_ids(root)
    n = max(ids) + 1
    assert ids == list(range(n)), f"non-contiguous shard ids in {root}: {ids[:8]}..."
    idx = pd.read_parquet(root / "index.parquet", columns=["shard"])
    per = idx.groupby("shard").size()
    jobs = [(a, min(a + args.shards_per_job, n))
            for a in range(0, n, args.shards_per_job)]
    print(f"[plan] {root}: {n} shards, {len(idx):,} rows -> {len(jobs)} jobs "
          f"({args.shards_per_job} shards/job, stage1={args.stage1})")
    for a, b in jobs:
        nrows = int(per.loc[a:b - 1].sum())
        out = out_path(Path(args.out_dir), args.part_tag or root.name,
                       args.stage1, a, b)
        done = " [DONE]" if out.exists() else ""
        print(f"[plan]   shards {a:3d}:{b:<3d} {nrows:9,} rows -> {out.name}{done}")
        print(f"sbatch --export=ALL,CMD='python 161_sweep_score.py "
              f"--cutout-root {root} --shard-range {a}:{b} --stage1 {args.stage1} "
              f"--out-dir {args.out_dir}"
              + (f" --part-tag {args.part_tag}" if args.part_tag else "")
              + (f" --ckpt-dir {args.ckpt_dir}" if args.ckpt_dir != str(C.DATA) else "")
              + (f" --lean-ckpt-dir {args.lean_ckpt_dir}"
                 if args.stage1 == "lean"
                 and args.lean_ckpt_dir != str(C.DATA / 'v2' / 'ckpt_lean') else "")
              + (f" --student-ckpt-dir {args.student_ckpt_dir}"
                 if args.stage1 == "student"
                 and args.student_ckpt_dir != str(C.DATA / 'v2' / 'ckpt_student') else "")
              + f"' nersc/shared_gpu.slurm")
    return 0


def write_sidecar(out: Path, args, a: int, b: int, elapsed: float) -> dict:
    """<out>.json — per-range audit record (operator/audit trail; 162 audits
    coverage against the 160 manifests directly and does not read this)."""
    df = pd.read_parquet(out)
    cols = [c for c in df.columns if c not in ("row_id", "ok")]
    fin = {c: int(np.isfinite(df[c].to_numpy(np.float64)).sum()) for c in cols}
    side = {"cutout_root": str(args.cutout_root), "part_tag": args.part_tag
            or Path(args.cutout_root).name, "shard_range": [a, b],
            "stage1": args.stage1, "out": str(out), "n_rows": int(len(df)),
            "n_ok": int(df["ok"].sum()) if "ok" in df.columns else None,
            "columns": cols, "n_finite": fin, "elapsed_s": round(elapsed, 1)}
    sp = out.parent / (out.name + ".json")
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(side, indent=2))
    tmp.rename(sp)
    print(f"[161] sidecar {sp.name}: {len(df):,} rows x {len(cols)} stage-1 cols")
    return side


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout-root", required=True,
                    help="ONE 111 part root (cutouts_<k>.npy + index.parquet)")
    ap.add_argument("--out-dir", required=True,
                    help="stage-1 score parquets dir (shared across jobs is SAFE: "
                         "file names encode part/mode/range)")
    ap.add_argument("--stage1", required=True, choices=("members", "lean", "student"),
                    help="stage-1 scorer set (the 150 gate's pick; 'lean' = the "
                         "LEAN v2 roster via 112 --only-extra on --lean-ckpt-dir)")
    ap.add_argument("--shard-range", default=None, metavar="A:B",
                    help="half-open shard range of this job (default: ALL shards)")
    ap.add_argument("--part-tag", default=None,
                    help="label in the output file name (default: cutout-root basename)")
    ap.add_argument("--ckpt-dir", default=str(C.DATA),
                    help="112 --ckpt-dir (members mode roster checkpoints)")
    ap.add_argument("--lean-ckpt-dir", default=str(C.DATA / "v2" / "ckpt_lean"),
                    help="dir with EXACTLY the five v2lean member_<name>.pt "
                         "(lean mode; 112 --only-extra --extra-ckpt-dir)")
    ap.add_argument("--student-ckpt-dir", default=str(C.DATA / "v2" / "ckpt_student"),
                    help="dir with member_student_distilled.pt (student mode; "
                         "112 --only-extra --extra-ckpt-dir)")
    ap.add_argument("--batch", type=int, default=512, help="112 --batch")
    ap.add_argument("--plan", action="store_true",
                    help="print the sbatch fan-out for this part root and exit")
    ap.add_argument("--shards-per-job", type=int, default=8,
                    help="--plan: shards per shared-GPU job")
    args = ap.parse_args()
    t0 = time.time()
    if args.plan:
        return plan(args)

    root, out_dir = Path(args.cutout_root), Path(args.out_dir)
    assert (root / "index.parquet").exists(), f"{root}: no index.parquet (run 111 first)"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = shard_ids(root)
    a, b = parse_range(args.shard_range) if args.shard_range else (0, max(ids) + 1)
    in_range = [s for s in ids if a <= s < b]
    if not in_range:
        print(f"[161] FATAL: shard range {a}:{b} matches none of {len(ids)} shards")
        return 1
    part_tag = args.part_tag or root.name
    out = out_path(out_dir, part_tag, args.stage1, a, b)
    print(f"[161] part={part_tag} stage1={args.stage1} shards {a}:{b} "
          f"({len(in_range)} present) -> {out}")

    if out.exists():        # 112 writes atomically -> existence == complete
        print(f"[161] {out.name} already exists -> resume: skipping the 112 pass")
    else:
        cmd = [sys.executable, str(C.ROOT / "112_score_pool.py"),
               "--cutout-root", str(root), "--out", str(out),
               "--shard-range", f"{a}:{b}", "--batch", str(args.batch)]
        if args.stage1 == "members":
            cmd += ["--ckpt-dir", args.ckpt_dir, "--skip-baselines"]
        elif args.stage1 == "lean":
            ld = Path(args.lean_ckpt_dir)
            stems = sorted(p.stem for p in ld.glob("member_*.pt")
                           if not p.name.endswith("_smoke.pt"))
            want = sorted(STAGE1_COLS["lean"])
            assert stems == want, (
                f"--lean-ckpt-dir {ld}: member_*.pt stems {stems} != the v2lean "
                f"roster {want} — place exactly those five checkpoints there "
                f"(112 --only-extra scores EVERY member_*.pt and names columns "
                f"by stem; the persisted v2lean fits expect those columns)")
            cmd += ["--ckpt-dir", args.ckpt_dir,
                    "--only-extra", "--extra-ckpt-dir", str(ld)]
        else:
            sd = Path(args.student_ckpt_dir)
            assert list(sd.glob("member_*.pt")), \
                f"--student-ckpt-dir {sd}: no member_*.pt (run 150 first)"
            cmd += ["--ckpt-dir", args.ckpt_dir,
                    "--only-extra", "--extra-ckpt-dir", str(sd)]
        print(f"[161] exec: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        assert out.exists(), f"112 returned 0 but {out} is missing"

    side = write_sidecar(out, args, a, b, time.time() - t0)
    want = STAGE1_COLS[args.stage1]
    if want is not None:
        missing = [c for c in want if c not in side["columns"]]
        assert not missing, f"{out.name}: missing stage-1 columns {missing}"
    print(f"[161] done ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
