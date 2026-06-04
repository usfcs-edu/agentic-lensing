#!/usr/bin/env python3
"""Run the spectroscopic VI pre-grader over the gold spectroscopic set and score it.

Gold set:
  * Hsu 2025 Table-2 Grade-A (20)         — discordant-z pairs, truth = lens (positives)
  * Foundry-II confirmed (20)             — spectroscopically confirmed, truth = lens
  * Foundry-II non-lens (4)               — confirmed NON-lenses, truth = not_lens (neg)

Acceptance (plan §5.5): re-grade >=18/20 Hsu Grade-A as plausible AND reject >=3/4
Foundry-II non-lenses.

  python lensjudge/spectro/run.py --concurrency 5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.spectro import grader  # noqa: E402
from lensjudge.tools.spectrum import sis_theta_e  # noqa: E402


def build_set() -> list[dict]:
    import json
    pairs = []
    # Hsu Table-2 Grade-A — join matches to classified_pairs for full features
    xm = config.HSU_DATA / "xmatch_table2.json"
    cls_p = config.HSU_DATA / "classified_pairs.parquet"
    if xm.exists() and cls_p.exists():
        matches = json.load(open(xm))["matches"]
        cp = pd.read_parquet(cls_p).set_index("group_id")
        for m in matches:
            gid = m["nearest_group_id"]
            row = cp.loc[gid] if gid in cp.index else None
            mz = sorted(m.get("member_z", [0, 0]))
            pairs.append(dict(
                name=m["name"], ra=m["hsu_ra"], dec=m["hsu_dec"],
                z_lens=float(row["Z_lens"]) if row is not None else mz[0],
                z_src=float(row["Z_src"]) if row is not None else mz[-1],
                sep_arcsec=float(row["sep_arcsec"]) if row is not None else None,
                sigma_v_lens=float(row["sigma_v_lens"]) if row is not None else None,
                theta_E_arcsec=float(row["theta_E_arcsec"]) if row is not None else None,
                logmstar_lens=float(row["logmstar_lens"]) if row is not None else None,
                class_algo=str(row["class_algo"]) if row is not None else "conventional",
                truth="lens", source="hsu_tab2"))
    # Foundry-II confirmed + non-lens
    g = config.FOUNDRY_II_DATA / "foundry_ii_master_comparison.csv"
    if g.exists():
        fii = pd.read_csv(g)
        for _, r in fii[fii.section.isin(["confirmed", "nonlens"])].iterrows():
            zl, zs, sv = r.get("z_lens_pub"), r.get("z_source_pub"), r.get("sigma_v_pub")
            pairs.append(dict(
                name=str(r["name"]).replace(" ", "_"), ra=r["ra_deg"], dec=r["dec_deg"],
                z_lens=zl, z_src=zs, sep_arcsec=None, sigma_v_lens=sv,
                theta_E_arcsec=sis_theta_e(sv, zl, zs), logmstar_lens=None,
                class_algo="conventional",
                truth="lens" if r["section"] == "confirmed" else "not_lens",
                source="foundry_ii_" + r["section"]))
    return pairs


async def run(pairs, concurrency, model):
    sem = asyncio.Semaphore(concurrency)
    trace_dir = config.OUT / "traces_spectro"; trace_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    async def one(p):
        async with sem:
            res = await grader.grade(p, model=model,
                                     trace_path=str(trace_dir / f"{p['name']}.jsonl"))
        g = res.grade
        rows.append({**{k: p[k] for k in ("name", "truth", "source", "z_lens", "z_src",
                                          "sigma_v_lens", "sep_arcsec")},
                     "parse_ok": res.parse_ok, "cost_usd": res.cost_usd,
                     "cls": g.cls if g else None, "plausible": g.plausible if g else None,
                     "confidence": g.confidence if g else None,
                     "rationale": (g.rationale if g else res.raw[:200])})
    await asyncio.gather(*(one(p) for p in pairs))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    print(f"\n[spectro] {len(df)} graded | parse_ok {int(df.parse_ok.sum())}/{len(df)} "
          f"| mean cost ${df.cost_usd.mean():.3f}")
    hsu = df[df.source == "hsu_tab2"]
    if len(hsu):
        plaus = ((hsu.plausible == True) & (hsu.cls != "not_lens")).sum()  # noqa: E712
        print(f"  Hsu Table-2 Grade-A: {plaus}/{len(hsu)} graded plausible lens "
              f"(target >=18/20)  [cls: {hsu.cls.value_counts().to_dict()}]")
    non = df[df.source == "foundry_ii_nonlens"]
    if len(non):
        rej = ((non.cls == "not_lens") | (non.plausible == False)).sum()  # noqa: E712
        print(f"  Foundry-II non-lenses: {rej}/{len(non)} rejected (target >=3/4)")
    conf = df[df.source == "foundry_ii_confirmed"]
    if len(conf):
        plaus = ((conf.plausible == True) & (conf.cls != "not_lens")).sum()  # noqa: E712
        print(f"  Foundry-II confirmed: {plaus}/{len(conf)} graded plausible lens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(config.OUT / "preds_spectro.csv"))
    args = ap.parse_args()
    pairs = build_set()
    print(f"[set] {len(pairs)} spectroscopic candidates: "
          f"{pd.Series([p['source'] for p in pairs]).value_counts().to_dict()}")
    df = asyncio.run(run(pairs, args.concurrency, args.model))
    df.to_csv(args.out, index=False)
    summarize(df)
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
