#!/usr/bin/env python3
"""Result-table statistics for the paper, computed from the run repo's CSVs.

Usage: python3 paper_stats.py /path/to/jwst-strong-lens-search

Prints a markdown digest (numbers quoted in the paper) and writes
analysis/out/paper_stats.json with every computed value.
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

def load(rel):
    with open(os.path.join(REPO, rel)) as f:
        return list(csv.DictReader(f))

S = {}

# ---- results.csv: verdicts, confidence, galaxy types, evidence terms --------
res = load("results/results.csv")
S["n_results_rows"] = len(res)
S["verdicts"] = dict(Counter(r["lens_at_center"] for r in res))
S["quadrant_flags"] = sum(1 for r in res if r["quadrant_lens"] not in ("", "none"))
S["galaxy_types_all"] = dict(Counter(r["center_galaxy_type"] for r in res))

conf = defaultdict(list)
for r in res:
    try:
        conf[r["lens_at_center"]].append(float(r["confidence"]))
    except ValueError:
        pass
S["confidence_by_verdict"] = {
    k: {"n": len(v), "min": min(v), "max": max(v),
        "mean": sum(v) / len(v),
        "median": sorted(v)[len(v) // 2]}
    for k, v in conf.items() if v}

# Evidence-term frequencies over all flagged rows (evidence non-empty).
TERMS = ["arc", "counter-image", "counter image", "ring", "knot", "blue",
         "concave", "tangential", "radial", "residual", "curved", "crescent",
         "companion", "spiral", "tidal", "shell", "merger", "edge-on",
         "diffraction", "spike", "over-subtraction", "colour contrast",
         "color contrast", "quad", "einstein"]
ev_rows = [r for r in res if (r.get("evidence") or "").strip()]
S["n_with_evidence"] = len(ev_rows)
tc = Counter()
for r in ev_rows:
    e = r["evidence"].lower()
    for t in TERMS:
        if t in e:
            tc[t] += 1
# merge spelling variants
tc["counter-image"] += tc.pop("counter image", 0)
tc["colour contrast"] += tc.pop("color contrast", 0)
S["evidence_term_rows"] = {t: n for t, n in tc.most_common()}

# ---- verification outcomes --------------------------------------------------
ver = load("results/verifications.csv")
S["n_verification_judgements"] = len(ver)
S["n_verified_candidates"] = len({r["id"] for r in ver})
by_persona = defaultdict(Counter)
for r in ver:
    by_persona[r["persona"]][r["verdict"]] += 1
S["verification_by_persona"] = {k: dict(v) for k, v in by_persona.items()}
alt = Counter()
ALT_CLASSES = {
    "spiral": "spiral arm/ring in host", "arm": "spiral arm/ring in host",
    "ring": "spiral arm/ring in host",
    "tidal": "tidal/merger debris", "merger": "tidal/merger debris",
    "shell": "tidal/merger debris", "stream": "tidal/merger debris",
    "companion": "chance alignment/companion",
    "chance": "chance alignment/companion",
    "background": "chance alignment/companion",
    "unrelated": "chance alignment/companion",
    "subtraction": "reduction/model artifact",
    "artifact": "reduction/model artifact", "artefact": "reduction/model artifact",
    "residual": "reduction/model artifact", "psf": "PSF/diffraction",
    "spike": "PSF/diffraction", "diffraction": "PSF/diffraction",
    "edge-on": "edge-on/inclined disk", "disk": "edge-on/inclined disk",
}
for r in ver:
    if r["verdict"] != "fail":
        continue
    a = (r["alternative"] or "").lower()
    hit = None
    for k, cls in ALT_CLASSES.items():
        if k in a:
            hit = cls
            break
    alt[hit or "other/unspecified"] += 1
S["fail_alternative_classes"] = dict(alt.most_common())

# ---- grades ------------------------------------------------------------------
S["grades_all"] = dict(Counter(r["grade"] for r in res if r["grade"]))

# ---- controls -----------------------------------------------------------------
ctl = load("results/control_recovery.csv")
S["n_controls"] = len(ctl)
S["controls_flagged"] = sum(1 for r in ctl if r["flagged"] == "True")
def er(r):
    try:
        return float(r["einstein_radius"])
    except ValueError:
        return None
for lab, lo, hi in [("lt0p75", 0.0, 0.75), ("0p75to1p2", 0.75, 1.2),
                    ("gt1p2", 1.2, 99.0)]:
    sub = [r for r in ctl if er(r) is not None and lo <= er(r) < hi]
    S[f"controls_theta_{lab}"] = {
        "n": len(sub),
        "flagged": sum(1 for r in sub if r["flagged"] == "True")}
S["controls_no_theta"] = {
    "n": sum(1 for r in ctl if er(r) is None),
    "flagged": sum(1 for r in ctl if er(r) is None and r["flagged"] == "True")}
best = [r for r in ctl if int(r["nA"]) >= 3]
S["controls_cowls_ge3A"] = {
    "n": len(best), "flagged": sum(1 for r in best if r["flagged"] == "True")}
S["controls_in_top100"] = sum(
    1 for r in ctl if r["rank"] and int(float(r["rank"])) <= 100)

# ---- known-lens completeness ---------------------------------------------------
kl = load("results/known_lens_recovery.csv")
S["n_known_on_cutouts"] = len(kl)
S["known_flagged"] = sum(1 for r in kl if r["flagged"] == "True")
gls = [r for r in kl if "gLS" in r["lens_src"]]
S["known_gLS"] = {"n": len(gls),
                  "flagged": sum(1 for r in gls if r["flagged"] == "True")}
S["known_src_breakdown"] = {
    src: {"n": len(v), "flagged": sum(1 for r in v if r["flagged"] == "True")}
    for src, v in
    ((s, [r for r in kl if r["lens_src"] == s])
     for s in sorted({r["lens_src"] for r in kl}))}
kn_conf = sorted((float(r["confidence"]) for r in kl if r["flagged"] == "True"),
                 reverse=True)
S["known_flagged_conf_max"] = kn_conf[0] if kn_conf else None

# ---- top-100 master ------------------------------------------------------------
top = load("results/JWST_top100_master.csv")
S["top100_grades"] = dict(Counter(r["verifier_grade"] for r in top))
S["top100_status"] = dict(Counter(r["discovery_status"] for r in top))
S["top100_with_zspec"] = sum(1 for r in top if r["deflector_z_spec"].strip())
S["top100_with_zphot"] = sum(1 for r in top if r["deflector_z_phot"].strip())
S["top100_programmes"] = len({r["jwst_programme"] for r in top})
S["top100_sw_filters"] = dict(Counter(r["sw_filter"] for r in top))

# ---- einstein radii -------------------------------------------------------------
try:
    era = load("results/einstein_radii.csv")
    vals = []
    for r in era:
        for k in ("theta_e", "theta_E", "einstein_radius", "theta_e_arcsec"):
            if k in r and r[k].strip():
                try:
                    vals.append(float(r[k]))
                except ValueError:
                    pass
                break
    if vals:
        vals.sort()
        S["theta_E_n"] = len(vals)
        S["theta_E_median"] = vals[len(vals) // 2]
        S["theta_E_min"] = vals[0]
        S["theta_E_max"] = vals[-1]
except FileNotFoundError:
    pass

with open(os.path.join(OUT, "paper_stats.json"), "w") as f:
    json.dump(S, f, indent=2)

for k, v in S.items():
    print(f"{k}: {v}")
