#!/usr/bin/env python
"""
Proportional stratified sub-manifest builder for the SpectrumFM scaling ladder.

The full DR1 mix manifest (`manifest_mix.jsonl`) is a list of JSONL records, one
per healpix coadd, each carrying {survey, program, healpix, coadd, redrock,
n_rows}. The scaling ladder's DATA axis needs smaller manifests that are FAITHFUL
miniatures of the full mix: same four-way survey×program proportions, and — so the
healpix-disjoint train/val split (`split_records_by_healpix`) keeps behaving — the
subsample must be taken BY RECORD (whole healpix coadds), never by row. Splitting a
single healpix across train and val would leak; keeping records whole preserves the
disjoint-split guarantee at every ladder rung.

Method: PROPORTIONAL stratified subsample by record. The strata are the distinct
(survey, program) pairs. Within each stratum we seeded-shuffle the records and keep
the first round(frac * len(stratum)) of them. Because the keep-fraction is applied
per stratum, the survey/program MIX is preserved (each stratum shrinks by ~frac);
because we keep whole records, the healpix-disjoint split still partitions cleanly.
Determinism: a per-stratum seed derived from (global seed, stratum key) means the
same seed -> identical record set, and the choice is independent of stratum
iteration order.

The summary prints per-stratum kept/total record counts, the realized record
fraction and n_rows (spectra) fraction vs the original, and confirms both land
near `frac`.

Usage:
  ~/.venvs/redshifty/bin/python tools/spectrumfm/make_submanifest.py \
      --frac 0.25 --out /raid/benson/data/desi_dr1_medium/manifest_mix_25pct.jsonl

  # determinism / proportion smoke (writes to /tmp, no GPU):
  ~/.venvs/redshifty/bin/python tools/spectrumfm/make_submanifest.py --smoke
"""
import argparse
import hashlib
import json
import random
import sys
import tempfile
from collections import OrderedDict, defaultdict
from pathlib import Path

DEFAULT_MANIFEST = "/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl"


def load_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratum_key(rec):
    """The stratum a record belongs to: its (survey, program) pair."""
    return (str(rec.get("survey", "")), str(rec.get("program", "")))


def _stratum_seed(seed, key):
    """Deterministic per-stratum seed from (global seed, stratum key).

    Hashing the key (rather than using the global seed directly for every
    stratum, or relying on dict-iteration order) makes each stratum's shuffle
    independent and order-invariant: the kept set depends only on (seed, key,
    members), not on how many strata were processed before it.
    """
    h = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest()
    return int(h[:16], 16)


def subsample(records, frac, seed):
    """Proportional stratified-by-record subsample.

    Returns (kept_records, stats) where kept_records preserves the input order
    (records are reordered only within the per-stratum shuffle used to *pick*
    them, then re-emitted in original manifest order for a stable, diffable file)
    and stats is an OrderedDict keyed by stratum with per-stratum counts.
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"--frac must be in (0, 1], got {frac}")

    # Group record INDICES by stratum, preserving first-seen stratum order for
    # a stable summary table.
    strata = OrderedDict()
    for i, rec in enumerate(records):
        strata.setdefault(stratum_key(rec), []).append(i)

    keep_idx = set()
    stats = OrderedDict()
    for key, idxs in strata.items():
        rng = random.Random(_stratum_seed(seed, key))
        shuffled = idxs[:]            # copy; do not mutate the grouping
        rng.shuffle(shuffled)
        k = round(frac * len(idxs))
        chosen = shuffled[:k]
        keep_idx.update(chosen)
        n_rows_tot = sum(int(records[i].get("n_rows", 0)) for i in idxs)
        n_rows_kept = sum(int(records[i].get("n_rows", 0)) for i in chosen)
        stats[key] = dict(
            n_records=len(idxs), kept_records=k,
            n_rows=n_rows_tot, kept_rows=n_rows_kept,
        )

    # Re-emit in original manifest order (stable / diff-friendly).
    kept = [records[i] for i in range(len(records)) if i in keep_idx]
    return kept, stats


def print_summary(stats, n_records_orig, n_rows_orig, frac):
    kept_rec = sum(s["kept_records"] for s in stats.values())
    kept_rows = sum(s["kept_rows"] for s in stats.values())
    hdr = f"  {'survey/program':<18} {'recs':>7} {'kept':>7} {'rec%':>7} {'rows':>9} {'krows':>9} {'row%':>7}"
    print(f"[submanifest] target frac = {frac:.4f}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, s in stats.items():
        name = f"{key[0]}/{key[1]}"
        rpct = 100.0 * s["kept_records"] / s["n_records"] if s["n_records"] else 0.0
        wpct = 100.0 * s["kept_rows"] / s["n_rows"] if s["n_rows"] else 0.0
        print(f"  {name:<18} {s['n_records']:>7d} {s['kept_records']:>7d} "
              f"{rpct:>6.1f}% {s['n_rows']:>9d} {s['kept_rows']:>9d} {wpct:>6.1f}%")
    print("  " + "-" * (len(hdr) - 2))
    rec_frac = kept_rec / n_records_orig if n_records_orig else 0.0
    row_frac = kept_rows / n_rows_orig if n_rows_orig else 0.0
    print(f"  {'TOTAL':<18} {n_records_orig:>7d} {kept_rec:>7d} "
          f"{100*rec_frac:>6.1f}% {n_rows_orig:>9d} {kept_rows:>9d} {100*row_frac:>6.1f}%")
    print(f"[submanifest] realized record fraction = {rec_frac:.4f}  "
          f"(target {frac:.4f}); realized n_rows (spectra) fraction = {row_frac:.4f}")
    return rec_frac, row_frac


def write_jsonl(records, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def run(args):
    records = load_records(args.manifest)
    n_records_orig = len(records)
    n_rows_orig = sum(int(r.get("n_rows", 0)) for r in records)
    print(f"[submanifest] read {n_records_orig} records / {n_rows_orig} spectra "
          f"from {args.manifest}")
    kept, stats = subsample(records, args.frac, args.seed)
    print_summary(stats, n_records_orig, n_rows_orig, args.frac)
    write_jsonl(kept, args.out)
    print(f"[submanifest] wrote {len(kept)} records -> {args.out}")


# ----------------------------------------------------------------------------
# CPU smoke — no GPU, no FITS I/O: proportions + determinism on the real
# manifest (or a synthetic one if the real manifest is absent).
# ----------------------------------------------------------------------------
def run_smoke(args):
    print("[SMOKE] CPU only — proportional-subsample correctness + determinism.")
    src = args.manifest
    if not Path(src).exists():
        print(f"[SMOKE] manifest {src} absent; using a synthetic 4-stratum manifest.")
        rng = random.Random(0)
        synth = []
        plan = [("sv3", "bright", 300), ("sv3", "dark", 250),
                ("main", "bright", 200), ("main", "dark", 150)]
        for surv, prog, n in plan:
            for h in range(n):
                synth.append(dict(survey=surv, program=prog, healpix=h,
                                  coadd=f"{surv}-{prog}-{h}", redrock=f"rr-{h}",
                                  n_rows=rng.randint(1, 4000)))
        src = str(Path(tempfile.gettempdir()) / "smoke_manifest_mix.jsonl")
        write_jsonl(synth, src)

    records = load_records(src)
    n0 = len(records)
    rows0 = sum(int(r.get("n_rows", 0)) for r in records)
    orig_strata = defaultdict(int)
    for r in records:
        orig_strata[stratum_key(r)] += 1

    frac = 0.25
    kept, stats = subsample(records, frac, args.seed)
    rec_frac, row_frac = print_summary(stats, n0, rows0, frac)

    # (1) ~25% of records overall.
    assert abs(rec_frac - frac) < 0.02, f"record fraction {rec_frac} off target {frac}"
    print(f"[SMOKE] PASS overall record fraction {rec_frac:.4f} ~ {frac}")

    # (2) per-stratum proportion within a couple % of the original mix.
    for key, s in stats.items():
        orig_share = orig_strata[key] / n0
        kept_total = sum(v["kept_records"] for v in stats.values())
        kept_share = s["kept_records"] / kept_total if kept_total else 0.0
        assert abs(kept_share - orig_share) < 0.02, (
            f"stratum {key} mix drifted: kept {kept_share:.3f} vs orig {orig_share:.3f}")
    print("[SMOKE] PASS per-stratum mix preserved within 2% of original")

    # (3) deterministic across two runs with the same seed.
    kept_a, _ = subsample(records, frac, args.seed)
    kept_b, _ = subsample(records, frac, args.seed)
    sig = lambda recs: [r.get("coadd") for r in recs]
    assert sig(kept_a) == sig(kept_b), "non-deterministic with fixed seed"
    print("[SMOKE] PASS deterministic record set across two runs (same seed)")

    # (4) a different seed yields a different (but same-size) set.
    kept_c, _ = subsample(records, frac, args.seed + 1)
    same_size = len(kept_c) == len(kept_a)
    differs = sig(kept_c) != sig(kept_a)
    print(f"[SMOKE] seed sensitivity: size match={same_size}, set differs={differs}")
    assert same_size and differs, "seed change should give same-size, different set"
    print("[SMOKE] OK.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--frac", type=float, default=0.25,
                    help="target keep fraction in (0, 1]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output jsonl path")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU proportions+determinism smoke (no output written to --out)")
    args = ap.parse_args()
    if args.smoke:
        run_smoke(args)
        return
    if args.out is None:
        ap.error("--out is required (or use --smoke)")
    run(args)


if __name__ == "__main__":
    main()
