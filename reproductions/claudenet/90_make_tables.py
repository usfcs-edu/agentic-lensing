#!/usr/bin/env python3
"""90_make_tables.py — aggregate every phase's result file into one master summary
(data/results_summary.json + a printed markdown table). Robust to missing files
(phases still running are shown as pending).

    /home2/benson/.venvs/claudenet/bin/python 90_make_tables.py
"""
from __future__ import annotations

import json

import pandas as pd

import _clib as C

D = C.DATA


def jload(name):
    p = D / name
    return json.load(open(p)) if p.exists() else None


def main():
    S = {}

    # baseline + flagship
    base = jload("meta_metrics_staged.json")
    fv = jload("flagship_verdict.json")
    if base and fv:
        r = base["recovery_at_fpr"]
        S["baseline_meta"] = {
            "storfer@1%": r["storfer|meta|0.01"]["recovery"],
            "storfer@0.1%": r["storfer|meta|0.001"]["recovery"],
            "inchausti@1%": r["inchausti|meta|0.01"]["recovery"],
            "inchausti@0.1%": r["inchausti|meta|0.001"]["recovery"]}
        S["flagship"] = {"verdict": fv["verdict"],
                         "wins_vs_meta": f"{fv['wins_vs_baseline_meta']}/{fv['n_metrics']}",
                         "wins_vs_best_member": f"{fv['wins_vs_best_member']}/{fv['n_metrics']}",
                         "storfer@1%": fv["ref"]["storfer_1"]["best_learned"],
                         "storfer@0.1%": fv["ref"]["storfer_01"]["best_learned"],
                         "inchausti@1%": fv["ref"]["inchausti_1"]["best_learned"],
                         "inchausti@0.1%": fv["ref"]["inchausti_01"]["best_learned"]}

    div = jload("diversity.json")
    if div:
        S["diversity"] = {"members": div["members"],
                          "pearson_offdiag_mean": round(div["pearson_offdiag_mean"], 3),
                          "spearman_offdiag_mean": round(div["spearman_offdiag_mean"], 3)}

    gate = jload("gate_phase0.json")
    if gate:
        S["phase0_gate"] = {"verdict": gate["verdict"],
                            "pearson_aion_effnet": round(gate["correlation"]["pearson"], 3),
                            "spearman": round(gate["correlation"]["spearman"], 3)}

    mine = jload("mining_summary.json")
    if mine:
        S["phase2_mining"] = {"verdict": mine["verdict"],
                              "hard_minus_random_storfer@1%": round(mine["hard_minus_random_storfer_1"], 3),
                              "results": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                                          for k, v in mine["results"].items()}}

    conf = jload("conformal_selection.json")
    if conf:
        S["phase4_conformal"] = {"average": conf["average"]}

    uq = jload("uncertainty.json")
    if uq:
        S["phase6_uncertainty"] = {"best_member": uq["best_member"],
                                   "selective": uq.get("selective"), "ood": uq.get("ood")}

    da = jload("domain_adapt.json")
    if da:
        S["phase5_domain_adapt"] = da

    eq = jload("equivariance.json")
    if eq:
        S["phase7_equivariance"] = eq

    le = D / "label_efficiency.csv"
    if le.exists():
        df = pd.read_csv(le)
        piv = df[df.cat == "storfer"].pivot(index="n_pos", columns="method", values="rec_1")
        S["phase3_label_efficiency"] = {"storfer@1%_by_npos": piv.round(3).to_dict()}

    (D / "results_summary.json").write_text(json.dumps(S, indent=2))

    print("# ClaudeNet — results summary\n")
    if "flagship" in S:
        print("## Phase 1 flagship vs published meta-learner (recovery @ matched FPR)\n")
        print("| metric | published meta | ClaudeNet | delta |")
        print("|---|---|---|---|")
        for m, key in (("Storfer @1%FPR", "storfer@1%"), ("Storfer @0.1%FPR", "storfer@0.1%"),
                       ("Inchausti @1%FPR", "inchausti@1%"), ("Inchausti @0.1%FPR", "inchausti@0.1%")):
            b = S["baseline_meta"][key]; f = S["flagship"][key]
            print(f"| {m} | {b:.3f} | {f:.3f} | {f-b:+.3f} |")
        print(f"\nVerdict: **{S['flagship']['verdict']}** "
              f"(beats published meta on {S['flagship']['wins_vs_meta']} metrics)")
    print("\nphases present:", [k for k in S])
    print("[90] wrote results_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
