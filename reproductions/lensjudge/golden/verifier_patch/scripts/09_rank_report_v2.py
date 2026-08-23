"""Stage 9 (v2): validate every verdict, aggregate the evidence-first scheme, rank, report.

Loads EVERY results/verdicts/verify_*.jsonl - the `*_ctl*` control files included (the v1
verifications.csv export predates them: 12 COWLS controls examined 0/3 were reported as U,
"never examined"). Each line is validated with plain dict checks against the record
contracts (advocate / critic / arbitrator / legacy pass-fail); anything else goes to
results/verdicts_rejected.jsonl with a reason and is NEVER silently coerced to
"uncertain". The script asserts that every verdict id lands in results_v2.csv.

Scoring (scripts/aggregate_v2.py, byte-identical to lensjudge's):
  S      = p_evidence * prod_i (1 - r_i * a_i)   over critics that named an alternative,
           a_i = the fraction of the advocate's items the critic's location box actually covers
  S_arb  = the same over critics the arbitrator upheld / partially upheld (secondary; equal to
           S when no arbitrator ran)
  letter = thresholds (scripts/thresholds_v2.json, --model-key) + the D rule on S, without the
           arbitrator (lensjudge's grade_pred); letter_arb = the same on S_arb with the
           arbitrator's rulings; letter_source says whether the thresholds were calibrated for
           this model key, are the Sonnet design-half numbers (uncalibrated here) or provisional
  rank   = examined items by S (ties: p_evidence, inspector confidence); U = never examined
           under v2, STRICTLY below every examined item (score_v2 = -1 + confidence/100)

Outputs: results/results_v2.csv, results/top100_v2.csv, results/verifications_v2.jsonl
(every accepted record, verbatim, with its source file and line), results/verdicts_rejected.jsonl,
results/regrade_diff.csv (v1 grade <-> v2 letter per id), results/report_v2.md

  python scripts/09_rank_report_v2.py [--model-key opus_claude_code] [--top-n 100]
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import util  # noqa: E402
import aggregate_v2  # noqa: E402

BASE = util.BASE
HERE = os.path.dirname(os.path.abspath(__file__))
CRITICS = ("artifact", "geometry", "morphology")
CRITERIA = ("source_contrast", "low_surface_brightness", "curvature", "counter_image", "arc_morphology")
PANELS = {"a", "b", "c", "d", "e", "f", "ctx"}
SCALES = {"galaxy", "group", "cluster", "none"}
ALTERNATIVES = {"spiral_arm", "ring_galaxy", "shell_tidal", "merger", "edge_on_disk",
                "companion_projection", "star_forming_clump", "diffraction_spike",
                "detector_artifact", "subtraction_residual", "psf_wing", "scale_tension", "other"}
NO_OPINION_REASONS = {"outside_competence", "feature_not_in_my_views", "image_quality"}
LETTERS = {"A", "B", "C", "D"}
RULINGS = {"upheld", "partial", "overruled"}
ORD = {"A": 3, "B": 2, "C": 1, "D": 0}
ADVOCATE_KEYS = {"id", "persona", "criteria", "items", "arc_radius_arcsec", "arc_pa_span_deg",
                 "counter_image_pos", "centre_of_curvature_offset_arcsec", "scale_class",
                 "n_red_neighbours_10as", "bcg_like_halo", "deflector_is_centre", "p_evidence",
                 "nothing_because", "notes"}
ITEM_KEYS = {"k", "what", "panel", "r_arcsec", "pa_deg_from", "pa_deg_to", "visible_in_direct", "criteria"}
CRITIC_KEYS = {"id", "persona", "no_opinion", "no_opinion_reason", "alternative", "alternative_desc",
               "location", "accounts_for", "leaves_standing", "refutation_strength", "measured",
               "scale_class", "notes"}
ARBITRATOR_KEYS = {"id", "persona", "rulings", "surviving_items", "letter_llm", "scale_class_final",
                   "needs_human", "rationale"}
LEGACY_KEYS = {"id", "persona", "verdict", "alternative", "notes"}


# ---------------------------------------------------------------- validation (plain dict checks)
def _num(x, lo=None, hi=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and math.isnan(x)):
        return False
    return (lo is None or x >= lo) and (hi is None or x <= hi)


def _int_list(x):
    return isinstance(x, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in x)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_advocate(d):
    """Plain-dict mirror of lensjudge's pydantic AdvocateRecord (golden/schemas_panel.py):
    the same record must be accepted, and scored the same, on both sides."""
    extra = set(d) - ADVOCATE_KEYS
    if extra:
        return f"unknown keys {sorted(extra)}"
    c = d.get("criteria")
    if not isinstance(c, dict) or set(c) != set(CRITERIA) or not all(_int(c[k]) and 0 <= c[k] <= 10 for k in CRITERIA):
        return "criteria must be the five INTEGER 0-10 scores"
    if not isinstance(d.get("items"), list):
        return "items must be a list"
    ks = []
    for it in d["items"]:
        if not isinstance(it, dict) or set(it) - ITEM_KEYS or not (ITEM_KEYS - {"criteria"}) <= set(it):
            return "item keys"
        crit = it.get("criteria", [])                       # lensjudge default: []
        if not (_int(it["k"]) and it["k"] >= 1 and isinstance(it["what"], str) and it["panel"] in PANELS
                and _num(it["r_arcsec"], 0) and _num(it["pa_deg_from"]) and _num(it["pa_deg_to"])
                and isinstance(it["visible_in_direct"], bool) and _int_list(crit)
                and all(1 <= v <= 5 for v in crit)):
            return f"item {it.get('k')} field types"
        ks.append(it["k"])
    if len(set(ks)) != len(ks):
        return "item k must be unique"
    if d.get("scale_class") not in SCALES:
        return "scale_class"
    if not _num(d.get("p_evidence"), 0, 1):
        return "p_evidence must be 0-1"
    for k in ("bcg_like_halo", "deflector_is_centre"):
        if k in d and not isinstance(d[k], bool):
            return f"{k} must be bool"
    if "n_red_neighbours_10as" in d and not (isinstance(d["n_red_neighbours_10as"], int) and not isinstance(d["n_red_neighbours_10as"], bool)):
        return "n_red_neighbours_10as must be int"
    for k in ("nothing_because", "notes"):
        if k in d and not isinstance(d[k], str):
            return f"{k} must be str"
    return None


def validate_critic(d):
    """Plain-dict mirror of lensjudge's CriticRecord: "" reads as null for the enums; a null
    refutation_strength is 0 when no alternative is named (and REQUIRED when one is); a named
    alternative needs a location box; scale_tension is capped at 0.4."""
    extra = set(d) - CRITIC_KEYS
    if extra:
        return f"unknown keys {sorted(extra)}"
    if not isinstance(d.get("no_opinion"), bool):
        return "no_opinion must be bool"
    for k in ("alternative", "no_opinion_reason"):
        if isinstance(d.get(k), str) and d[k].strip() == "":
            d[k] = None
    if d.get("no_opinion_reason") is not None and d["no_opinion_reason"] not in NO_OPINION_REASONS:
        return "no_opinion_reason"
    alt = d.get("alternative")
    if alt is not None and alt not in ALTERNATIVES:
        return f"alternative {alt!r}"
    if d["no_opinion"] and alt is not None:
        return "no_opinion with a named alternative"
    if d.get("refutation_strength") is None:
        if alt is not None:
            return f"alternative {alt!r} needs a refutation_strength"
        d["refutation_strength"] = 0.0
    if not _num(d.get("refutation_strength"), 0, 1):
        return "refutation_strength must be 0-1"
    if alt == "scale_tension" and d["refutation_strength"] > 0.4:
        return "scale_tension strength > 0.4"
    loc = d.get("location")
    if loc is not None and not (isinstance(loc, dict)
                                and set(loc) == {"r_arcsec_from", "r_arcsec_to", "pa_deg_from", "pa_deg_to"}
                                and all(_num(v) for v in loc.values())
                                and _num(loc["r_arcsec_from"], 0) and _num(loc["r_arcsec_to"], 0)):
        return "location box"
    if alt is not None and loc is None:
        return f"alternative {alt!r} needs a location box"
    for k in ("accounts_for", "leaves_standing"):
        if not _int_list(d.get(k, [])):
            return f"{k} must be a list of item numbers"
    if d.get("measured") is not None and not isinstance(d["measured"], dict):
        return "measured"
    if d.get("scale_class") is not None and not isinstance(d["scale_class"], str):
        return "scale_class"
    return None


def validate_arbitrator(d):
    extra = set(d) - ARBITRATOR_KEYS
    if extra:
        return f"unknown keys {sorted(extra)}"
    if not isinstance(d.get("rulings"), list):
        return "rulings must be a list"
    personas = []
    for r in d["rulings"]:
        if not (isinstance(r, dict) and r.get("persona") in CRITICS and r.get("ruling") in RULINGS
                and _int_list(r.get("covers", [])) and isinstance(r.get("why", ""), str)):
            return "ruling"
        personas.append(r["persona"])
    if len(set(personas)) != len(personas):
        return "one ruling per critic"
    if not _int_list(d.get("surviving_items", [])):
        return "surviving_items"
    if d.get("letter_llm") not in LETTERS:
        return f"letter_llm {d.get('letter_llm')!r}"
    if "needs_human" in d and not isinstance(d["needs_human"], bool):
        return "needs_human"
    return None


def validate_legacy(d):
    if set(d) - LEGACY_KEYS:
        return f"unknown keys {sorted(set(d) - LEGACY_KEYS)}"
    if str(d.get("verdict", "")).lower().strip() not in {"pass", "fail", "uncertain"}:
        return f"verdict {d.get('verdict')!r}"
    return None


def classify(d):
    """-> (kind, error) with kind in advocate|critic|arbitrator|legacy|None."""
    if not isinstance(d, dict) or "id" not in d or not str(d.get("id", "")).strip():
        return None, "no id"
    p = str(d.get("persona", "")).lower().strip()
    if p == "advocate":
        return "advocate", validate_advocate(d)
    if p == "arbitrator":
        return "arbitrator", validate_arbitrator(d)
    if p in CRITICS:
        if "verdict" in d and "refutation_strength" not in d:
            return "legacy", validate_legacy(d)
        return "critic", validate_critic(d)
    return None, f"unknown persona {p!r}"


def load_verdicts(verdict_dir):
    """(accepted, rejected): accepted = [{file, line, kind, record}], rejected = [{file, line, reason, text}]."""
    acc, rej = [], []
    for f in sorted(glob.glob(os.path.join(verdict_dir, "verify_*.jsonl"))):
        for i, line in enumerate(open(f), start=1):
            raw = line.strip().rstrip(",")
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except Exception as e:   # noqa: BLE001
                rej.append({"file": os.path.basename(f), "line": i, "reason": f"json: {e}", "text": raw[:500]})
                continue
            kind, err = classify(d)
            if kind is None or err:
                rej.append({"file": os.path.basename(f), "line": i, "reason": err or "unclassified",
                            "id": d.get("id") if isinstance(d, dict) else None, "text": raw[:500]})
                continue
            acc.append({"file": os.path.basename(f), "line": i, "kind": kind, "record": d})
    return acc, rej


# ---------------------------------------------------------------- thresholds
UNCALIBRATED = "sonnet_thresholds_uncalibrated"


def resolve_thresholds(raw, model_key):
    """aggregate_v2.resolve_thresholds: the model's own block when calibrated
    (letter_source "<model_key>_calibrated"); else the Sonnet-API numbers frozen on the
    lensjudge design half, labelled UNCALIBRATED ("Sonnet-API design-half thresholds, not
    checked on this model"); else - before that freeze - the a-priori `provisional` block,
    labelled "provisional"."""
    thr = dict(aggregate_v2.resolve_thresholds(raw, model_key, fallback_keys=("sonnet_api", "provisional")))
    if thr["thresholds_key"] == "sonnet_api" and model_key != "sonnet_api":
        thr["letter_source"] = UNCALIBRATED
    return thr


def strongest_alternative(critics, arbitrator):
    """The named alternative with the largest r*a among critics not overruled (the critic the
    letter rule would cite); '' when no critic named one."""
    overruled = set()
    if arbitrator:
        overruled = {r["persona"] for r in arbitrator.get("rulings", []) if r.get("ruling") == "overruled"}
    best, best_v = "", -1.0
    for c in critics:
        if c.get("no_opinion") or not c.get("alternative") or c["persona"] in overruled:
            continue
        v = float(c["refutation_strength"])
        if v > best_v:
            best, best_v = c["alternative"], v
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-key", default="opus_claude_code",
                    help="thresholds_v2.json key for the model that produced the verdicts")
    ap.add_argument("--thresholds", default=os.path.join(HERE, "thresholds_v2.json"))
    ap.add_argument("--verdicts", default=f"{BASE}/results/verdicts")
    ap.add_argument("--inspections", default=f"{BASE}/results/inspections.csv")
    ap.add_argument("--results-v1", default=f"{BASE}/results/results.csv")
    ap.add_argument("--controls", default=f"{BASE}/data/controls.csv")
    ap.add_argument("--out-dir", default=f"{BASE}/results")
    ap.add_argument("--top-n", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    thr = resolve_thresholds(json.load(open(a.thresholds)), a.model_key)
    print(f"thresholds: {thr}", flush=True)
    acc, rej = load_verdicts(a.verdicts)
    with open(os.path.join(a.out_dir, "verdicts_rejected.jsonl"), "w") as fh:
        for r in rej:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out_dir, "verifications_v2.jsonl"), "w") as fh:
        for r in acc:
            fh.write(json.dumps(r) + "\n")
    kinds = collections.Counter(r["kind"] for r in acc)
    print(f"verdict lines: {len(acc)} accepted {dict(kinds)}; {len(rej)} rejected -> verdicts_rejected.jsonl", flush=True)

    # first record per (id, persona) wins (the v1 rule; replicates stay visible in verifications_v2.jsonl)
    adv, crit, arb, legacy = {}, collections.defaultdict(dict), {}, collections.defaultdict(dict)
    for r in acc:
        d = r["record"]
        cid, p = str(d["id"]), str(d["persona"]).lower()
        if r["kind"] == "advocate":
            adv.setdefault(cid, d)
        elif r["kind"] == "critic":
            crit[cid].setdefault(p, d)
        elif r["kind"] == "arbitrator":
            arb.setdefault(cid, d)
        else:
            legacy[cid].setdefault(p, d)
    verdict_ids = {str(r["record"]["id"]) for r in acc}

    ins = pd.read_csv(a.inspections)
    ins["id"] = ins["id"].astype(str)
    ins = ins.drop_duplicates("id").set_index("id", drop=False)
    v1 = None
    if os.path.exists(a.results_v1):
        v1 = pd.read_csv(a.results_v1, dtype={"id": str}).drop_duplicates("id").set_index("id")

    rows = []
    all_ids = list(ins.index) + sorted(verdict_ids - set(ins.index))
    for cid in all_ids:
        base = ins.loc[cid].to_dict() if cid in ins.index else {"id": cid}
        row = {"id": cid, "in_inspections": cid in ins.index}
        for c in ("ra", "dec", "mag_r", "type", "proposal", "field_id", "sw_filter", "lw_filter",
                  "lens_at_center", "quadrant_lens", "center_galaxy_type", "png"):
            row[c] = base.get(c)
        row["flagged"] = bool(base.get("flagged")) if base.get("flagged") == base.get("flagged") else False
        conf = base.get("confidence")
        row["confidence"] = float(conf) if conf is not None and conf == conf else 0.0
        a_rec = adv.get(cid)
        critics = [c for c in crit.get(cid, {}).values()]
        arb_rec = arb.get(cid)
        row["examined"] = a_rec is not None
        if a_rec is not None:
            # exactly lensjudge's schemas_panel.assemble: the PRIMARY letter is the unarbitrated
            # S under the unarbitrated guards; the arbitrator enters S_arb / letter_arb only
            # (S_arb == S when no arbitrator ran: no critic named an alternative)
            S = float(aggregate_v2.score_S(a_rec, critics))
            S_arb = float(aggregate_v2.score_S_arb(a_rec, critics, arb_rec))
            letter, source = aggregate_v2.assign_letter(S, a_rec, critics, thr)
            letter_arb, _ = aggregate_v2.assign_letter(S_arb, a_rec, critics, thr, arbitrator=arb_rec)
            row.update({
                "S": S,
                "S_arb": S_arb,
                "p_evidence": float(a_rec["p_evidence"]),
                "letter": letter, "letter_arb": letter_arb, "letter_source": source,
                "letter_llm": arb_rec["letter_llm"] if arb_rec else "",
                "scale_class_final": arb_rec.get("scale_class_final") if arb_rec else a_rec.get("scale_class"),
                "scale_class_advocate": a_rec.get("scale_class"),
                "deflector_is_centre": a_rec.get("deflector_is_centre"),
                "alternative_final": strongest_alternative(critics, arb_rec),
                "n_items": len(a_rec["items"]),
                "n_critics": len(critics),
                "n_no_opinion": sum(1 for c in critics if c.get("no_opinion")),
                "n_named": sum(1 for c in critics if not c.get("no_opinion") and c.get("alternative")),
                "arbitrated": arb_rec is not None,
                "needs_human": bool(arb_rec.get("needs_human")) if arb_rec else False,
                "nothing_because": a_rec.get("nothing_because", ""),
                "advocate_notes": str(a_rec.get("notes", ""))[:300],
                "arbitrator_rationale": str(arb_rec.get("rationale", ""))[:600] if arb_rec else "",
                **{f"crit_{k}": a_rec["criteria"][k] for k in CRITERIA},
            })
        else:
            row.update({"S": np.nan, "S_arb": np.nan, "p_evidence": np.nan, "letter": "U" if row["flagged"] else "",
                        "letter_arb": "", "letter_source": "", "letter_llm": "", "scale_class_final": "", "scale_class_advocate": "",
                        "deflector_is_centre": None, "alternative_final": "", "n_items": 0, "n_critics": 0,
                        "n_no_opinion": 0, "n_named": 0, "arbitrated": False, "needs_human": False,
                        "nothing_because": "", "advocate_notes": "", "arbitrator_rationale": "",
                        **{f"crit_{k}": np.nan for k in CRITERIA}})
        # the v1 view of the same id: recorded grade (results.csv) and the pass-count over every
        # legacy verdict line on disk (the ctl files included)
        leg = list(legacy.get(cid, {}).values())
        n_pass, n_fail, n_unc, pc_letter = aggregate_v2.passcount_incumbent(leg)
        row.update({"legacy_n_pass": n_pass, "legacy_n_fail": n_fail, "legacy_n_uncertain": n_unc,
                    "legacy_grade_from_jsonl": pc_letter if leg else ("U" if row["flagged"] else ""),
                    "legacy_grade_v1": (str(v1.at[cid, "grade"]) if v1 is not None and cid in v1.index
                                        and isinstance(v1.at[cid, "grade"], str) else "")})
        row["score_v2"] = row["S"] if row["examined"] else (-1.0 + row["confidence"] / 100.0)
        rows.append(row)
    df = pd.DataFrame(rows)

    # ranking: examined (has S) strictly above every U, then S desc, p_evidence desc, confidence desc
    df = df.sort_values(["examined", "S", "p_evidence", "confidence", "id"],
                        ascending=[False, False, False, False, True], na_position="last").reset_index(drop=True)
    # aggregate_v2.rank_key (ascending => best first) must agree with the explicit sort above
    # on the examined-above-U and S-desc structure (it breaks ties by id, the sort by confidence)
    keys = [aggregate_v2.rank_key(r) for r in df.to_dict("records")]
    if [k[:2] for k in keys] != sorted(k[:2] for k in keys):
        print("WARNING: aggregate_v2.rank_key orders the table differently from the explicit sort", flush=True)
    df["rank_v2"] = np.arange(1, len(df) + 1)
    assert (df.loc[df["examined"], "rank_v2"].max() if df["examined"].any() else 0) < \
        (df.loc[~df["examined"], "rank_v2"].min() if (~df["examined"]).any() else len(df) + 1), \
        "an unexamined item outranks an examined one"
    # every verdict id on disk must land in a row that SHOWS it (examined under v2, or with
    # its legacy pass-count from the jsonl) - not merely exist in the table by construction
    shown = set(df.loc[df["examined"] | (df["legacy_n_pass"] + df["legacy_n_fail"] + df["legacy_n_uncertain"] > 0), "id"])
    missing = verdict_ids - shown
    assert not missing, f"{len(missing)} verdict ids are absent from results_v2: {sorted(missing)[:10]}"

    cols = ["rank_v2", "id", "ra", "dec", "mag_r", "type", "proposal", "field_id", "sw_filter", "lw_filter",
            "flagged", "confidence", "examined", "S", "S_arb", "p_evidence", "letter", "letter_arb", "letter_source",
            "letter_llm", "scale_class_final", "scale_class_advocate", "deflector_is_centre",
            "alternative_final", "n_items", "n_critics", "n_no_opinion", "n_named", "arbitrated",
            "needs_human", "score_v2", "legacy_grade_v1", "legacy_grade_from_jsonl", "legacy_n_pass",
            "legacy_n_fail", "legacy_n_uncertain", "lens_at_center", "quadrant_lens", "center_galaxy_type",
            "nothing_because", "advocate_notes", "arbitrator_rationale", "in_inspections", "png"] \
        + [f"crit_{k}" for k in CRITERIA]
    df[cols].to_csv(os.path.join(a.out_dir, "results_v2.csv"), index=False)
    cand = df[df["flagged"] | df["examined"]].copy()
    cand.head(a.top_n)[cols].to_csv(os.path.join(a.out_dir, "top100_v2.csv"), index=False)
    ex = df[df["examined"]]
    diff = ex[["id", "legacy_grade_v1", "legacy_grade_from_jsonl", "legacy_n_pass", "letter", "letter_arb", "letter_llm",
               "S", "S_arb", "p_evidence", "alternative_final", "rank_v2"]].copy()
    diff["delta_ordinal"] = [ORD.get(n, np.nan) - ORD.get(o, np.nan) if o in ORD and n in ORD else np.nan
                             for o, n in zip(diff["legacy_grade_v1"], diff["letter"])]
    diff.to_csv(os.path.join(a.out_dir, "regrade_diff.csv"), index=False)

    # ---- report
    lines = ["# JWST strong-lens verification v2 (evidence-first)", "",
             f"- verdict lines accepted: **{len(acc)}** {dict(kinds)}; rejected: **{len(rej)}** "
             f"(`results/verdicts_rejected.jsonl`)",
             f"- items examined under v2: **{len(ex)}**; flagged never examined (U, ranked below every examined item): "
             f"**{int((df['flagged'] & ~df['examined']).sum())}**",
             f"- thresholds: `{thr}` -> every letter carries `letter_source={thr['letter_source']}`",
             f"- letters: {ex['letter'].value_counts().to_dict()}; arbitrated: {int(ex['arbitrated'].sum())}; "
             f"needs_human: {int(ex['needs_human'].sum())}",
             f"- no_opinion votes: {int(ex['n_no_opinion'].sum())} of {int(ex['n_critics'].sum())} critic records",
             ""]
    if len(ex) and (ex["legacy_grade_v1"] != "").any():
        ct = pd.crosstab(ex["legacy_grade_v1"].replace("", "none"), ex["letter"])
        lines += ["## v1 grade (rows) vs v2 letter (columns)", "```", ct.to_string(), "```", ""]
    ctl_path = a.controls
    if os.path.exists(ctl_path):
        ctl = pd.read_csv(ctl_path, dtype={"id": str})
        j = ctl.merge(df[["id", "examined", "letter", "S", "rank_v2", "legacy_grade_v1"]], on="id", how="left")
        jx = j[j["examined"].fillna(False).astype(bool)]
        lines += ["## Positive controls (data/controls.csv)",
                  f"- examined under v2: {len(jx)}/{len(j)}; at A/B: **{int(jx['letter'].isin(['A', 'B']).sum())}**; "
                  f"at A/B/C: {int(jx['letter'].isin(['A', 'B', 'C']).sum())}; in top {a.top_n}: "
                  f"{int((jx['rank_v2'] <= a.top_n).sum())}", ""]
    top = cand.head(20)
    lines += [f"## Top {min(20, len(top))}", "",
              "| rank | id | S | letter | letter_llm | scale | alternative_final | v1 grade |", "|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        s = f"{r['S']:.3f}" if r["S"] == r["S"] else "U"
        lines.append(f"| {r['rank_v2']} | {r['id']} | {s} | {r['letter']} | {r['letter_llm']} | "
                     f"{r['scale_class_final']} | {r['alternative_final']} | {r['legacy_grade_v1']} |")
    open(os.path.join(a.out_dir, "report_v2.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]), flush=True)
    print(f"results_v2.csv: {len(df)} rows; top{a.top_n}_v2 written; regrade_diff: {len(diff)} rows", flush=True)
    print("REPORT_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
