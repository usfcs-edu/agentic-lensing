#!/usr/bin/env python3
"""132_probe_gate_v2.py — Phase 130: the AION-upgrade decision gate (runs
LOCALLY, one GPU, claudenet venv).

Question: do NATIVE-griz AION embeddings (160px dr10 griz cutouts, real i band
where it exists) beat the v1 DEGRADED member (101px grz -> bilinear 160 +
synthetic i=(r+z)/2) by enough to justify the Phase-130 LoRA fine-tune — and
on which variant?

GATE (a) is MATCHED-ROWS and IN-EXPERIMENT. The stored v1 number
(flagship_operating_point.csv row 'aion': storfer@1%=0.6137) came from a probe
trained on the 9,373-row v1 trainpool at a different neg:pos ratio than the
full staged split — comparing a full-train native probe against it is
confounded. Instead, TWO probes are trained with the IDENTICAL recipe and
IDENTICAL training rows (the v1 trainpool row_ids from --gate-train-manifest):
  probe_degraded  on the v1 degraded embeddings (--degraded-emb-root:
      aion_emb_trainpool_base.npy, (9373,768), aligned 1:1 with that manifest)
  probe_native    on the new native-base embeddings restricted to those SAME
      row_ids (a subset of the union manifest — coverage asserted; any row
      without a usable native embedding is reported and dropped from BOTH)
Both early-stop on the same v1 val rows (aion_in_val_manifest.parquet +
aion_emb_val_base.npy vs the native embeddings of those rows), then
  gate_a = native storfer@1% >= degraded-refit storfer@1% + --gate-margin
with both recoveries on each probe's own OLD-testneg thresholds
(_ensemble.recovery_at_fpr — the exact 28_eval_flagship arithmetic). The
stored 0.6137 is printed as a REFERENCE LINE ONLY (sanity: the refit should
land near it; the diff is printed and warned about if |diff| > 0.05).

Per --variants entry the full-train "member probe" is unchanged: the v1-recipe
MLP probe (arch + training copied from 12_probe_aion.py: MLPProbe hidden=256
dropout=0.1, xmu/xsd standardisation fit on the TRAIN rows, AdamW lr 1e-3 wd
1e-4, cosine T_max=epochs, batch 256, class weight [1, n0/n1], early stop
patience 25 on val AUC, seed C.SEED) trained on the FULL v1 staged TRAIN split
(labels from training_split_staged.parquet), early-stopped on the staged VAL
split, scoring val/testneg/storfer/inchausti — it feeds Phase 140 and keeps
writing scores_member_aion_native_<variant>.parquet. Reported per variant:
  * standalone recovery@1%/0.1%FPR on the OLD v1 testneg (v1-comparable),
  * Pearson/Spearman vs the v1 effnet_S2 member's raw scores per split
    (decorrelation check, scores_member_effnet_S2.parquet),
  * the with-i vs without-i recovery split (north rows have i_ok=False and a
    zero i plane behind their embeddings; they are REPORTED, never dropped).
Only the GATE arithmetic uses the matched-rows probe pair.

Inputs (131_embed_aion_variants.py outputs, rsynced back from Perlmutter):
--emb-root, a COMMA-LIST of roots (the 131 outputs land as two: south +
north), concatenated via each root's emb_<variant>_index.parquet (combined
row_id uniqueness asserted); per root and variant,
  emb_<variant>.npy             (N_ok, dim) float16 mean-pooled AION embeddings
  emb_<variant>_index.parquet   one row per manifest object; columns row_id,
                                ok, i_ok, nan_frac, emb_row (-1 = no
                                embedding). A 1:1 layout (len(npy)==len(index),
                                no emb_row) also works.
plus the v1 degraded artefacts for the gate:
  --gate-train-manifest  data/emb/aion_in_trainpool_manifest.parquet
                         (row_id, label, fits_path, i_synth)
  --degraded-emb-root    data/emb: aion_emb_trainpool_base.npy +
                         aion_emb_{val,testneg,storfer,inchausti}_base.npy,
                         each aligned 1:1 with data/emb/aion_in_<sp>_manifest
                         .parquet.

Outputs:
  data/v2/scores_member_aion_native_<variant>.parquet  v1 member-scores schema
      [split,row_id,label,p] over val/testneg/storfer/inchausti (so Phase 140
      can calibrate + refit the ensemble); p=NaN where the embedding is
      missing/not-ok/non-finite (113-style self-exclusion).
  data/v2/ckpt/aion_native_probe_<variant>.pt          probe head + xmu/xsd
  data/v2/aion_gate_v2.json                            all metrics + GATES:
      (a) matched-rows: gate_a_native >= gate_a_degraded_refit + --gate-margin
          (0.03) else verdict KEEP-V1-MEMBER (lora_justified=False); fields
          gate_a_native / gate_a_degraded_refit / gate_a_pass /
          degraded_stored_ref;
      (b) lora_variant = 'large' if storfer@1%(large) - storfer@1%(base) >=
          --large-margin (0.02) else 'base' (full-train probes);
      xlarge vs large delta is reported (no gate hangs off it).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      /home2/benson/.venvs/claudenet/bin/python 132_probe_gate_v2.py \
        --emb-root data/v2/aion_native_south,data/v2/aion_native_north \
        --variants base,large,xlarge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

import _clib as C
import _ensemble as E

V2 = C.DATA / "v2"
EVAL_SPLITS = ("val", "testneg", "storfer", "inchausti")
DEGRADED_FALLBACK = {"storfer_1": 0.614, "storfer_01": 0.289,
                     "inchausti_1": 0.766, "inchausti_01": 0.436}


class MLPProbe(nn.Module):
    """copied verbatim from 12_probe_aion.MLPProbe (= _probe.MLPHead, 2-class)."""

    def __init__(self, dim, hidden=256, k=2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, ytr, Xva, yva, epochs, lr, device):
    """12_probe_aion training recipe, with the staged VAL split (not an internal
    80/20 cut) as the early-stop set. Returns (head, best_val_auc)."""
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    Xv = torch.from_numpy(Xva).to(device)
    cw = torch.tensor([1.0, float((ytr == 0).sum() / max((ytr == 1).sum(), 1))],
                      device=device)
    head = MLPProbe(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=cw)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), 256):
            b = perm[s:s + 256]
            opt.zero_grad(); lossf(head(Xt[b]), yt[b]).backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            pv = torch.softmax(head(Xv), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    head.load_state_dict(best_state); head.eval()
    return head, float(best_auc)


def load_split_tables():
    """The v1 eval discipline tables: staged train/val (labels) + the three
    held-out eval manifests."""
    staged = pd.read_parquet(C.DATA / "training_split_staged.parquet")
    tabs = {"train": staged[staged.split == "train"][["row_id", "label"]]}
    for sp, f in (("val", "eval_val"), ("testneg", "eval_testneg"),
                  ("storfer", "eval_storfer"), ("inchausti", "eval_inchausti")):
        tabs[sp] = pd.read_parquet(C.DATA / f"{f}.parquet")[["row_id", "label"]]
    return tabs


def degraded_reference():
    """v1 degraded-AION member row from flagship_operating_point.csv."""
    f = C.DATA / "flagship_operating_point.csv"
    if f.exists():
        df = pd.read_csv(f)
        row = df[df.scorer == "aion"]
        if len(row):
            r = row.iloc[0]
            return {k: float(r[k]) for k in
                    ("storfer_1", "storfer_01", "inchausti_1", "inchausti_01")}
    print(f"[132] WARNING: {f} missing row 'aion' -> hardcoded fallback")
    return dict(DEGRADED_FALLBACK)


def gather(emb, pos_of, ok, finite, tab):
    """Map a split table to (X, y, found_mask, ok_mask) over its rows."""
    idx = tab.row_id.map(pos_of)
    found = idx.notna().to_numpy()
    iarr = idx[found].astype(int).to_numpy()
    usable = ok[iarr] & finite[iarr]
    return iarr, found, usable


def load_embeddings(roots: list[Path], variant: str):
    """Concatenate emb_<variant>.npy across the --emb-root entries (south +
    north 131 runs) into one NaN-holed (N, dim) float32 array + a combined
    index (row_id, ok [, i_ok if EVERY root provides it])."""
    embs, idxs, dim = [], [], None
    for root in roots:
        emb_f, idx_f = root / f"emb_{variant}.npy", root / f"emb_{variant}_index.parquet"
        raw = np.load(emb_f).astype(np.float32)      # fp16 on disk -> fp32 (v1 load)
        index = pd.read_parquet(idx_f)
        ok = (index["ok"].to_numpy(bool) if "ok" in index.columns
              else np.ones(len(index), bool))
        if "emb_row" in index.columns:               # 131 layout: npy = ok rows only
            er = index["emb_row"].to_numpy(np.int64)
            ok = ok & (er >= 0)
            assert not len(er[er >= 0]) or er.max() < len(raw), \
                f"{variant}: emb_row exceeds {emb_f} ({len(raw)} rows)"
            emb = np.full((len(index), raw.shape[1]), np.nan, np.float32)
            emb[er >= 0] = raw[er[er >= 0]]
        else:                                        # 1:1 layout
            assert len(raw) == len(index), \
                f"{variant}: emb {len(raw)} != index {len(index)} and no emb_row column"
            emb = raw
        assert dim is None or emb.shape[1] == dim, \
            f"{variant}: dim {emb.shape[1]} != {dim} across --emb-root entries"
        dim = emb.shape[1]
        sub = pd.DataFrame({"row_id": index.row_id, "ok": ok})
        if "i_ok" in index.columns:
            sub["i_ok"] = index.i_ok.to_numpy(bool)
        embs.append(emb)
        idxs.append(sub)
        print(f"[132] {variant}: {root} -> {len(index):,} rows, {int(ok.sum()):,} ok")
    if not all("i_ok" in s.columns for s in idxs):
        idxs = [s.drop(columns="i_ok", errors="ignore") for s in idxs]
    index = pd.concat(idxs, ignore_index=True)
    assert index.row_id.is_unique, \
        f"{variant}: duplicate row_id across the --emb-root entries"
    return np.concatenate(embs, 0), index


def eval_variant(variant, args, tabs, eff, emb, index, device):
    ok = index["ok"].to_numpy(bool)
    finite = np.isfinite(emb).all(1)
    i_ok = index["i_ok"].to_numpy(bool) if "i_ok" in index.columns else None
    pos_of = pd.Series(np.arange(len(index)), index=index.row_id)
    print(f"[132] {variant}: emb {emb.shape} ok={int(ok.sum()):,} "
          f"finite={int(finite.sum()):,} i_ok="
          f"{'n/a' if i_ok is None else f'{int(i_ok.sum()):,}'}")

    # --- train rows -> standardisation + probe -------------------------------
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    itr, ftr, utr = gather(emb, pos_of, ok, finite, tabs["train"])
    Xtr = emb[itr[utr]]
    ytr = tabs["train"].label.to_numpy()[ftr][utr].astype(np.int64)
    iva, fva, uva = gather(emb, pos_of, ok, finite, tabs["val"])
    cov = {sp: float(gather(emb, pos_of, ok, finite, tabs[sp])[2].sum()
                     / max(len(tabs[sp]), 1))
           for sp in ("train",) + EVAL_SPLITS}
    print(f"[132] {variant}: train n={len(ytr):,} (pos={int(ytr.sum()):,}) "
          f"coverage={{{', '.join(f'{k}:{v:.3f}' for k, v in cov.items())}}}")

    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-6        # 12_probe_aion standardisation
    Xva = ((emb[iva[uva]] - xmu) / xsd).astype(np.float32)
    yva = tabs["val"].label.to_numpy()[fva][uva].astype(np.int64)
    head, val_auc = train_probe(((Xtr - xmu) / xsd).astype(np.float32), ytr,
                                Xva, yva, args.epochs, args.lr, device)
    print(f"[132] {variant}: probe val AUC={val_auc:.4f} (dim={emb.shape[1]})")
    (V2 / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "xmu": xmu, "xsd": xsd,
                "dim": int(emb.shape[1]), "variant": variant, "val_auc": val_auc},
               V2 / "ckpt" / f"aion_native_probe_{variant}.pt")

    # --- score the four shared splits (NaN where unusable; never dropped) ----
    def score_rows(iarr, usable):
        p = np.full(len(usable), np.nan, np.float32)
        if usable.sum():
            Xs = torch.from_numpy(((emb[iarr[usable]] - xmu) / xsd)
                                  .astype(np.float32)).to(device)
            with torch.no_grad():
                p[usable] = torch.softmax(head(Xs), 1)[:, 1].cpu().numpy()
        return p

    out, split_p = [], {}
    for sp in EVAL_SPLITS:
        tab = tabs[sp].reset_index(drop=True)
        iarr, found, usable = gather(emb, pos_of, ok, finite, tab)
        p = np.full(len(tab), np.nan, np.float32)
        p[found] = score_rows(iarr, usable)
        d = tab.copy(); d["split"] = sp; d["p"] = p
        out.append(d[["split", "row_id", "label", "p"]])
        split_p[sp] = d
        print(f"[132] {variant}: scored {sp:10s} n={len(d):,} "
              f"finite={int(np.isfinite(p).sum()):,} mean_p={np.nanmean(p):.3f}")
    scores = pd.concat(out, ignore_index=True)
    sf = V2 / f"scores_member_aion_native_{variant}.parquet"
    scores.to_parquet(sf, index=False)
    print(f"[132] {variant}: wrote {sf}")

    # --- standalone recovery on the OLD v1 testneg (28_eval_flagship math) ---
    neg = split_p["testneg"].p.to_numpy()
    rec = {}
    for cat in ("storfer", "inchausti"):
        r = E.recovery_at_fpr(neg, split_p[cat].p.to_numpy(), fprs=C.TARGET_FPR)
        rec[cat] = {str(f): {"threshold": r[f]["threshold"],
                             "recovery": r[f]["recovery"], "n": r[f]["n_cand"]}
                    for f in C.TARGET_FPR}
        print(f"[132] {variant}: {cat:9s} recovery@1%={rec[cat]['0.01']['recovery']:.3f} "
              f"@0.1%={rec[cat]['0.001']['recovery']:.3f}")

    # --- decorrelation vs v1 effnet_S2 (raw p, per split + pooled) -----------
    corr, pooled = {}, []
    for sp in EVAL_SPLITS:
        m = split_p[sp].merge(eff[eff.split == sp][["row_id", "p"]]
                              .rename(columns={"p": "p_eff"}), on="row_id")
        m = m[np.isfinite(m.p) & np.isfinite(m.p_eff)]
        if len(m) >= 3:
            corr[sp] = {"pearson": float(pearsonr(m.p, m.p_eff)[0]),
                        "spearman": float(spearmanr(m.p, m.p_eff)[0]), "n": len(m)}
            pooled.append(m[["p", "p_eff"]])
    mp = pd.concat(pooled, ignore_index=True)
    corr["pooled"] = {"pearson": float(pearsonr(mp.p, mp.p_eff)[0]),
                      "spearman": float(spearmanr(mp.p, mp.p_eff)[0]), "n": len(mp)}
    print(f"[132] {variant}: corr vs effnet_S2 pooled pearson="
          f"{corr['pooled']['pearson']:.3f} spearman={corr['pooled']['spearman']:.3f}"
          + "".join(f" | {sp} r_s={corr[sp]['spearman']:.3f}"
                    for sp in EVAL_SPLITS if sp in corr))

    # --- with-i vs without-i recovery split (north rows REPORTED, not dropped)
    i_split = None
    if i_ok is not None:
        i_split = {}
        thr = {f: E.fpr_threshold(neg, f) for f in C.TARGET_FPR}
        ntn = split_p["testneg"].row_id.map(pos_of).dropna().astype(int)
        i_split["testneg_i_ok_frac"] = float(i_ok[ntn.to_numpy()].mean())
        for cat in ("storfer", "inchausti"):
            d = split_p[cat].copy()
            ic = d.row_id.map(pos_of).map(
                lambda j: bool(i_ok[int(j)]) if pd.notna(j) else False)
            i_split[cat] = {}
            for flag, name in ((True, "with_i"), (False, "without_i")):
                p = d.p[ic == flag].to_numpy()
                p = p[np.isfinite(p)]
                i_split[cat][name] = {
                    "n": int(len(p)),
                    **{str(f): float((p >= thr[f]).mean()) if len(p) else None
                       for f in C.TARGET_FPR}}
            w, wo = i_split[cat]["with_i"], i_split[cat]["without_i"]
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(f"[132] {variant}: {cat:9s} i-split @1%: with_i={fmt(w['0.01'])} "
                  f"(n={w['n']}) without_i={fmt(wo['0.01'])} (n={wo['n']})")
    else:
        print(f"[132] {variant}: WARNING no i_ok column in the emb index -> "
              f"with-i/without-i split not reportable")

    return {"dim": int(emb.shape[1]), "n_emb": len(emb), "coverage": cov,
            "n_train": int(len(ytr)), "probe_val_auc": val_auc,
            "recovery": rec, "corr_vs_effnet_S2": corr, "i_split": i_split,
            "scores_parquet": str(sf)}


def matched_gate(args, tabs, emb, index, device):
    """GATE (a), matched rows + in-experiment: refit the DEGRADED probe and
    train the NATIVE base probe with the identical recipe on the identical v1
    trainpool rows, early-stop both on the same v1 val rows, then storfer/
    inchausti recovery on each probe's own OLD-testneg thresholds (the exact
    28_eval_flagship arithmetic)."""
    droot = Path(args.degraded_emb_root)
    ok = index["ok"].to_numpy(bool)
    finite = np.isfinite(emb).all(1)
    pos_of = pd.Series(np.arange(len(index)), index=index.row_id)

    def deg_split(sp):
        """v1 degraded embeddings + manifest for one split (aligned 1:1)."""
        m = pd.read_parquet(droot / f"aion_in_{sp}_manifest.parquet")
        m["row_id"] = m.row_id.astype(str)
        X = np.load(droot / f"aion_emb_{sp}_base.npy").astype(np.float32)
        assert len(X) == len(m), \
            f"gate: aion_emb_{sp}_base.npy {len(X)} rows != manifest {len(m)}"
        return m, X

    def native_rows(row_ids):
        """row_ids -> (union positions, found mask, usable mask)."""
        idx = row_ids.map(pos_of)
        found = idx.notna().to_numpy()
        ii = np.where(found, idx.fillna(0).to_numpy(), 0).astype(np.int64)
        return ii, found, found & ok[ii] & finite[ii]

    # identical training rows: the v1 trainpool, usable on BOTH sides ---------
    gtm = pd.read_parquet(args.gate_train_manifest)
    gtm["row_id"] = gtm.row_id.astype(str)
    Xd_tr = np.load(droot / "aion_emb_trainpool_base.npy").astype(np.float32)
    assert len(Xd_tr) == len(gtm), \
        f"gate: aion_emb_trainpool_base.npy {len(Xd_tr)} rows != " \
        f"{args.gate_train_manifest} {len(gtm)}"
    ti, tfound, tusable = native_rows(gtm.row_id)
    assert tfound.all(), \
        (f"gate: {int((~tfound).sum())} v1 trainpool row_ids missing from the "
         f"native union index, e.g. {gtm.row_id[~tfound].head(5).tolist()}")
    sel = tusable & np.isfinite(Xd_tr).all(1)
    if int((~sel).sum()):
        print(f"[gate] WARNING: {int((~sel).sum())}/{len(gtm)} trainpool rows "
              f"lack a usable embedding on one side -> dropped from BOTH "
              f"probes (training rows stay identical), e.g. "
              f"{gtm.row_id[~sel].head(5).tolist()}")
    ytr = gtm.label.to_numpy()[sel].astype(np.int64)

    # identical early-stop rows: the v1 val split, usable on BOTH sides -------
    vm, Xd_va = deg_split("val")
    vi, _, vusable = native_rows(vm.row_id)
    vsel = vusable & np.isfinite(Xd_va).all(1)
    yva = vm.label.to_numpy()[vsel].astype(np.int64)
    print(f"[gate] matched rows: train {int(sel.sum()):,}/{len(gtm):,} "
          f"(pos={int(ytr.sum()):,}) val {int(vsel.sum()):,}/{len(vm):,}")

    # the probe pair: identical recipe + rows, per-input xmu/xsd --------------
    feats = {"degraded": (Xd_tr[sel], Xd_va[vsel]),
             "native": (emb[ti[sel]], emb[vi[vsel]])}
    rec, aucs = {}, {}
    for tag, (Xt, Xv) in feats.items():
        torch.manual_seed(C.SEED); np.random.seed(C.SEED)
        xmu, xsd = Xt.mean(0), Xt.std(0) + 1e-6      # 12_probe_aion standardisation
        head, vauc = train_probe(((Xt - xmu) / xsd).astype(np.float32), ytr,
                                 ((Xv - xmu) / xsd).astype(np.float32), yva,
                                 args.epochs, args.lr, device)
        aucs[tag] = vauc

        def score(X):
            Xs = torch.from_numpy(((X - xmu) / xsd).astype(np.float32)).to(device)
            with torch.no_grad():
                return torch.softmax(head(Xs), 1)[:, 1].cpu().numpy()

        sp_p = {}
        for sp in ("testneg", "storfer", "inchausti"):
            if tag == "degraded":
                _, X = deg_split(sp)
                X = X[np.isfinite(X).all(1)]
            else:
                ii, _, u = native_rows(tabs[sp].row_id)
                X = emb[ii[u]]
            sp_p[sp] = score(X)
        rec[tag] = {}
        for cat in ("storfer", "inchausti"):
            r = E.recovery_at_fpr(sp_p["testneg"], sp_p[cat], fprs=C.TARGET_FPR)
            rec[tag][cat] = {str(f): float(r[f]["recovery"]) for f in C.TARGET_FPR}
        print(f"[gate] probe_{tag}: val AUC={vauc:.4f} | "
              f"storfer@1%={rec[tag]['storfer']['0.01']:.3f} "
              f"@0.1%={rec[tag]['storfer']['0.001']:.3f} | "
              f"inchausti@1%={rec[tag]['inchausti']['0.01']:.3f} "
              f"(own old-testneg threshold, n_neg={len(sp_p['testneg']):,})")
    return {"n_train": int(sel.sum()), "n_train_pos": int(ytr.sum()),
            "n_train_dropped": int((~sel).sum()), "n_val": int(vsel.sum()),
            "val_auc": aucs, "recovery": rec,
            "gate_train_manifest": str(args.gate_train_manifest),
            "degraded_emb_root": str(args.degraded_emb_root)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb-root", default=str(V2 / "aion_native"),
                    help="COMMA-LIST of dirs with emb_<variant>.npy + "
                         "emb_<variant>_index.parquet (e.g. the south,north "
                         "131 outputs); concatenated via the index parquets")
    ap.add_argument("--variants", default="base,large,xlarge",
                    help="comma list; 'base' is required for the gates")
    ap.add_argument("--epochs", type=int, default=200)   # 12_probe_aion defaults
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate-train-manifest",
                    default=str(C.EMB / "aion_in_trainpool_manifest.parquet"),
                    help="v1 trainpool manifest (row_id,label,fits_path,"
                         "i_synth) — the matched gate's training rows")
    ap.add_argument("--degraded-emb-root", default=str(C.EMB),
                    help="dir with the v1 degraded aion_emb_*_base.npy + "
                         "aion_in_*_manifest.parquet")
    ap.add_argument("--gate-margin", type=float, default=0.03,
                    help="(a) matched-rows native must beat the degraded "
                         "refit storfer@1%% by this")
    ap.add_argument("--large-margin", type=float, default=0.02,
                    help="(b) large must beat base storfer@1%% by this for LoRA-on-large")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    roots = [Path(r.strip()) for r in args.emb_root.split(",") if r.strip()]
    V2.mkdir(parents=True, exist_ok=True)

    tabs = load_split_tables()
    eff = pd.read_parquet(C.DATA / "scores_member_effnet_S2.parquet")
    degr = degraded_reference()
    print(f"[132] degraded v1 AION stored reference: storfer@1%={degr['storfer_1']:.4f} "
          f"@0.1%={degr['storfer_01']:.3f} (reference only) | "
          f"variants={variants} roots={[str(r) for r in roots]} device={device}")

    res, base_emb = {}, None
    for v in variants:
        emb, index = load_embeddings(roots, v)
        res[v] = eval_variant(v, args, tabs, eff, emb, index, device)
        if v == "base":
            base_emb = (emb, index)

    # ---- GATES ---------------------------------------------------------------
    # (a) matched-rows pair (the full-train probes above feed Phase 140 only)
    mg = matched_gate(args, tabs, *base_emb, device) if base_emb else None
    gate_a_native = mg["recovery"]["native"]["storfer"]["0.01"] if mg else None
    gate_a_degraded = mg["recovery"]["degraded"]["storfer"]["0.01"] if mg else None
    gate_a = (gate_a_native >= gate_a_degraded + args.gate_margin) if mg else None
    stored = degr["storfer_1"]
    if mg:
        diff = gate_a_degraded - stored
        print(f"[132] degraded refit storfer@1%={gate_a_degraded:.3f} vs stored "
              f"v1 reference {stored:.4f} (diff {diff:+.3f})"
              + ("  *** WARNING: |diff| > 0.05 — the refit landed far from the "
                 "stored v1 number; check the gate inputs ***"
                 if abs(diff) > 0.05 else ""))

    s1 = {v: res[v]["recovery"]["storfer"]["0.01"]["recovery"] for v in variants}
    d_large = (s1["large"] - s1["base"]) if {"base", "large"} <= s1.keys() else None
    d_xlarge = (s1["xlarge"] - s1["large"]) if {"large", "xlarge"} <= s1.keys() else None
    lora_variant = "large" if (d_large is not None and d_large >= args.large_margin) else "base"
    lora_justified = bool(gate_a)
    verdict = ("INCOMPLETE" if gate_a is None
               else f"LORA-ON-{lora_variant.upper()}" if gate_a else "KEEP-V1-MEMBER")

    gates = {
        "gate_a_native": gate_a_native,
        "gate_a_degraded_refit": gate_a_degraded,
        "gate_a_pass": gate_a,
        "degraded_stored_ref": stored,
        "gate_a_rule": f"matched-rows native storfer@1% >= degraded refit + "
                       f"{args.gate_margin} (own old-testneg thresholds; the "
                       f"stored {stored:.4f} is a reference line only)",
        "degraded_reference": degr,
        "storfer_at_1pct_fulltrain": s1,
        "delta_large_minus_base": d_large,
        "delta_xlarge_minus_large": d_xlarge,
        "large_margin": args.large_margin,
        "lora_variant": lora_variant,
        "lora_justified": lora_justified,
        "verdict": verdict,
    }
    out = {"variants": res, "matched_gate": mg, "gates": gates,
           "config": {"emb_root": args.emb_root, "epochs": args.epochs,
                      "lr": args.lr, "seed": C.SEED,
                      "gate_train_manifest": args.gate_train_manifest,
                      "degraded_emb_root": args.degraded_emb_root}}
    gf = V2 / "aion_gate_v2.json"
    gf.write_text(json.dumps(out, indent=2))

    if mg:
        print(f"\n[132] GATE (a) matched-rows: native {gate_a_native:.3f} vs "
              f"degraded refit {gate_a_degraded:.3f} + {args.gate_margin} = "
              f"{gate_a_degraded + args.gate_margin:.3f} -> "
              f"{'PASS' if gate_a else 'FAIL'} "
              f"(stored v1 ref {stored:.4f}, reference only)")
    else:
        print("\n[132] GATE (a): N/A — 'base' not in --variants")
    if d_large is not None:
        print(f"[132] GATE (b) large-base={d_large:+.3f} (need >= "
              f"{args.large_margin:+.3f}) -> lora_variant={lora_variant}")
    if d_xlarge is not None:
        print(f"[132] report: xlarge-large={d_xlarge:+.3f} (no gate)")
    print(f"[132] VERDICT: {verdict} (lora_justified={lora_justified}) -> wrote {gf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
