#!/usr/bin/env python3
"""121_retrain_mined_members.py — Phase 120: retrain ONE v1 ensemble member with
mined negatives swapped in at FIXED total negative count (runs LOCALLY, one GPU
per member; pin with CUDA_VISIBLE_DEVICES from the caller).

Recipe fidelity: the member's v1 training table (data/member_<name>_train.parquet
from 19_build_member_subsets.py) and v1 hyperparameters are reused EXACTLY —
arch/timm-variant/aug_seed from data/members.json, epochs/batch/accum/lr/decay_ep
and the model factory imported by path from 20_train_member.py, training via
_train.train_supervised. The ONLY change is the negative set: --n-mine of the
member's bootstrap-negative TRAIN ROWS (positions, so duplicate bootstrap draws
count separately) are displaced UNIFORMLY AT RANDOM with seed 2026+boot_seed
(per-member deterministic, identical displacement for the hard and random
variants) and replaced by the mined set from 120/120b. Total negative count —
and hence the neg:pos ratio — is unchanged (the v1 fixed-count control
discipline); positives and the val split are untouched.

Writes:
  data/v2/member_<name>_<variant>_train.parquet   (the swapped table, audit)
  data/v2/ckpt/member_<name>_<variant>.pt         (same schema as 20_train_member:
        state_dict/arch/score_arch/mean/std/val_auc/shielded_cfg, + variant so
        112_score_pool.load_member_checkpoint needs no members.json entry)
  data/v2/scores_member_<name>_<variant>.parquet  [split,row_id,label,p,pc]
        (v1 scores_member_* schema; pc = isotonic fit on val, as 25_calibrate)
  A smoke run (--epochs override) appends '_smoke' to ALL THREE artifact names
  (train table, checkpoint, scores) so it can never overwrite/alias the
  production artifacts.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 121_retrain_mined_members.py \
        --member effnet_S2 --variant hard
    # table-only sanity (no GPU): add --build-only
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import _clib as C

V2 = C.DATA / "v2"
MEMBERS = ("effnet_S2", "effnet_B3", "resnet46_C")


def build_swapped_table(member: str, spec: dict, mined: pd.DataFrame,
                        n_mine: int) -> tuple[pd.DataFrame, dict]:
    """v1 member table with n_mine bootstrap-negative train rows displaced
    (seed 2026+boot_seed, positions without replacement) by the mined rows."""
    df = pd.read_parquet(C.DATA / f"member_{member}_train.parquet")
    neg_pos = np.where((df["split"] == "train") & (df["label"] == 0))[0]
    n_pos = int(((df["split"] == "train") & (df["label"] == 1)).sum())
    n_val = int((df["split"] == "val").sum())
    assert n_mine <= len(neg_pos), \
        f"n_mine {n_mine} > {len(neg_pos)} bootstrap negatives"
    assert len(mined) >= n_mine, f"mined manifest has {len(mined)} < n_mine {n_mine}"
    mined = mined.head(n_mine).copy()           # 120 order: hardest first
    clash = set(mined["row_id"]) & set(df["row_id"].astype(str))
    assert not clash, f"{len(clash)} mined row_ids collide with the v1 table " \
                      f"(MinePool should be brick-disjoint), e.g. {sorted(clash)[:3]}"

    rng = np.random.default_rng(C.SEED + int(spec["boot_seed"]))
    displaced = neg_pos[rng.choice(len(neg_pos), size=n_mine, replace=False)]
    kept = df.drop(index=df.index[displaced])

    # attach RA/DEC from the MinePool manifest (LensDataset ignores them; kept
    # for schema parity with the v1 table)
    man_f = V2 / "minepool_manifest.parquet"
    if man_f.exists() and not {"RA", "DEC"} <= set(mined.columns):
        man = pd.read_parquet(man_f)[["row_id", "RA", "DEC"]]
        man["row_id"] = man["row_id"].astype(str)
        mined = mined.merge(man, on="row_id", how="left")
    for col in ("RA", "DEC"):
        if col not in mined.columns:
            mined[col] = np.nan
    mined_rows = pd.DataFrame({
        "row_id": mined["row_id"].astype(str), "label": 0,
        "RA": mined["RA"], "DEC": mined["DEC"],
        "fits_dir": mined["fits_dir"].astype(str), "split": "train"})
    out = pd.concat([kept[["row_id", "label", "RA", "DEC", "fits_dir", "split"]],
                     mined_rows], ignore_index=True)

    n_neg_after = int(((out["split"] == "train") & (out["label"] == 0)).sum())
    assert n_neg_after == len(neg_pos), \
        f"negative count changed: {len(neg_pos)} -> {n_neg_after}"
    assert int(((out["split"] == "train") & (out["label"] == 1)).sum()) == n_pos
    assert int((out["split"] == "val").sum()) == n_val
    info = {"n_train_pos": n_pos, "n_train_neg": len(neg_pos), "n_val": n_val,
            "n_displaced": n_mine,
            "n_unique_neg_before": int(df.iloc[neg_pos]["row_id"].nunique()),
            "n_unique_neg_after": int(out[(out.split == "train")
                                          & (out.label == 0)]["row_id"].nunique()),
            "displacement_seed": C.SEED + int(spec["boot_seed"])}
    return out, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--member", required=True, choices=MEMBERS)
    ap.add_argument("--variant", required=True, choices=("hard", "random"),
                    help="which mined set replaces the bootstrap negatives")
    ap.add_argument("--mined-manifest", default=None,
                    help="120b manifest parquet (default "
                         "data/v2/mined_<variant>_fits_manifest.parquet)")
    ap.add_argument("--n-mine", type=int, default=10000,
                    help="how many bootstrap negatives to displace")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the v1 epoch count (SMOKE TESTS ONLY)")
    ap.add_argument("--build-only", action="store_true",
                    help="build + write the swapped table, no training (no GPU)")
    args = ap.parse_args()
    name = f"{args.member}_{args.variant}"
    # smoke runs (--epochs override) get '_smoke' artifact names so they can
    # never overwrite/alias the production table/checkpoint/scores
    art = name + ("_smoke" if args.epochs is not None else "")
    if art != name:
        print(f"[121] --epochs {args.epochs} override -> SMOKE artifact names: "
              f"{V2 / f'member_{art}_train.parquet'}, "
              f"{V2 / 'ckpt' / f'member_{art}.pt'}, "
              f"{V2 / f'scores_member_{art}.parquet'}")

    roster = {m["name"]: m for m in json.load(open(C.DATA / "members.json"))}
    spec = roster[args.member]
    arch, aug_seed = spec["arch"], spec["aug_seed"]

    mined_f = (args.mined_manifest if args.mined_manifest
               else V2 / f"mined_{args.variant}_fits_manifest.parquet")
    mined = pd.read_parquet(mined_f)
    mined["row_id"] = mined["row_id"].astype(str)
    dfm, info = build_swapped_table(args.member, spec, mined, args.n_mine)
    print(f"[swap] {name}: displaced {info['n_displaced']:,}/{info['n_train_neg']:,} "
          f"bootstrap neg rows (seed {info['displacement_seed']}) with "
          f"{args.variant.upper()} mined negs from {mined_f}; "
          f"unique negs {info['n_unique_neg_before']:,} -> {info['n_unique_neg_after']:,}; "
          f"pos {info['n_train_pos']:,} + val {info['n_val']:,} untouched")
    V2.mkdir(parents=True, exist_ok=True)
    table_f = V2 / f"member_{art}_train.parquet"
    dfm.to_parquet(table_f, index=False)
    print(f"[swap] wrote {table_f}")
    if args.build_only:
        print(f"[121] --build-only: stopping before training")
        return 0

    import torch
    import _train as T
    M20 = C._load("cn_121_m20", C.ROOT / "20_train_member.py")   # build_model+EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs else M20.EPOCHS[arch]
    score_arch = "efficientnet" if arch == "efficientnet" else "shielded"
    model = M20.build_model(arch, spec.get("variant"))
    n_params = sum(p.numel() for p in model.parameters())
    batch, accum = {"efficientnet": (128, 2), "dihedral": (32, 4)}.get(arch, (128, 1))
    print(f"[train] {name} arch={arch} params={n_params:,} epochs={epochs} "
          f"batch={batch} accum={accum} aug_seed={aug_seed} "
          f"n_train={(dfm.split == 'train').sum()}")
    t0 = time.time()
    model, val_auc, mean, std = T.train_supervised(
        model, arch, dfm, device, epochs=epochs, batch=batch, lr=1e-3,
        decay_ep=max(8, epochs // 3), accum=accum, aug_seed=aug_seed)
    print(f"[train] {name} best_val_auc={val_auc:.4f} ({(time.time() - t0) / 60:.1f}m)")

    ckpt_dir = V2 / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "score_arch": score_arch,
                "mean": mean.tolist(), "std": std.tolist(), "val_auc": val_auc,
                "shielded_cfg": C.CFG194 if arch in ("shielded", "dihedral") else None,
                "variant": spec.get("variant"),            # timm variant for 112
                "mining": {"member": args.member, "mining_variant": args.variant,
                           "mined_manifest": str(mined_f), **info}},
               ckpt_dir / f"member_{art}.pt")
    print(f"[train] saved {ckpt_dir / f'member_{art}.pt'}")

    # score the shared v1 eval manifests (20_train_member verbatim) + isotonic pc
    import _ensemble as E
    rows = []
    for sp in ("val", "testneg", "storfer", "inchausti"):
        d = pd.read_parquet(C.DATA / f"eval_{sp}.parquet").copy()
        d["p"] = T.score_df(model, score_arch, d, mean, std, device)
        d["split"] = sp
        rows.append(d[["split", "row_id", "label", "p"]])
        print(f"[score] {sp:10s} n={len(d)} mean_p={d['p'].mean():.3f}")
    sc = pd.concat(rows, ignore_index=True)
    sc["p"] = sc["p"].astype(np.float32)
    val = sc[sc.split == "val"]
    cal = E.make_calibrator("isotonic").fit(val["p"].to_numpy(), val["label"].to_numpy())
    sc["pc"] = cal.transform(sc["p"].to_numpy())
    out_f = V2 / f"scores_member_{art}.parquet"
    sc.to_parquet(out_f, index=False)
    print(f"[121] {name} done -> {out_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
