#!/usr/bin/env python
"""Build normalized ground-truth table for Euclid Q1 SLDE lens-modeling adjudications.

Sources:
  - arXiv:2503.15325 (SLDE-B) Tables 1 (cat A) and 2 (cat B): per-object Model = success/expert-verdict
  - arXiv:2503.15324 (SLDE-A) Sect. 7.2: aggregate modeling counts (488 attempted, 374 successful,
    315 judged lenses, 59 judged non-lens) -- per-object judgments NOT published
  - Zenodo 10.5281/zenodo.15025832: modeling_lens_mass.csv (successful models) + archive listings
    (lens.zip / unsuccess.zip / group.zip / recenter.zip) + q1_discovery_engine_lens_catalog.csv
"""
import os, re
import numpy as np
import pandas as pd

SCRATCH = "/private/tmp/claude-501/-Users-benson-sync-research-agentic-lensing/16c65f46-2248-454d-8454-61780a59a64c/scratchpad"
REPO = "/Users/benson/sync/research/agentic-lensing"
CAT_PATH = f"{REPO}/reproductions/euclid-q1/data/raw/q1_discovery_engine_lens_catalog.csv"
OUT_PATH = f"{REPO}/reproductions/lensjudge/parity/data/external/euclid_q1_modeling.csv"
TEX_B = f"{SCRATCH}/arxiv/2503.15325/main.tex"

# ---------------------------------------------------------------- SLDE-B tables
def parse_sldeb_tables():
    tex = open(TEX_B).read()
    rows = []
    for label, catname, tabno in [("tab:catA", "A", "Table 1"), ("tab:catB", "B", "Table 2")]:
        block = tex.split(r"\label{%s}" % label, 1)[1].split(r"\end{tabular}", 1)[0]
        for line in block.splitlines():
            if "EUCL" not in line or "&" not in line:
                continue
            parts = [p.strip() for p in line.rstrip("\\").split("&")]
            if len(parts) < 9:
                continue
            clean = lambda s: re.sub(r"\\phantom\{[^}]*\}|\\,|\$|\{|\}|\\", "", s).strip()
            name = clean(parts[0]).replace("EUCL", "EUCL ").replace("  ", " ").strip()
            ra, dec = float(clean(parts[1])), float(clean(parts[2]))
            z, sigma, score = clean(parts[3]), clean(parts[4]), float(clean(parts[5]))
            model = clean(parts[6]).replace(" ", "")
            te = clean(parts[7])
            te = None if te in ("--", "-") else float(te)
            rows.append(dict(name=name, ra=ra, dec=dec, z_lens=z, sigma_v=sigma,
                             vi_score=score, model=model, theta_e=te, cat=catname, table=tabno))
    return pd.DataFrame(rows)

# ------------------------------------------------------- Zenodo archive id sets
def ids_from_listing(path, prefix):
    ids = set()
    for line in open(path):
        m = re.match(rf"{prefix}/([^/]+)/", line.strip())
        if m:
            ids.add(m.group(1))
    return ids

def decode_coords(id_str):
    m = re.match(r"(\d+)_((NEG)?)(\d+)$", id_str)
    neg, digits = bool(m.group(3)), m.group(4)
    ra = int(digits[: len(digits) - 9]) / 1e7
    dec = int(digits[-9:]) / 1e7
    return ra, -dec if neg else dec

# ------------------------------------------------------------------------ main
sldeb = parse_sldeb_tables()
cat = pd.read_csv(CAT_PATH)
mass = pd.read_csv(f"{SCRATCH}/zenodo/modeling_lens_mass.csv").set_index("id_str")

zips = {z: ids_from_listing(f"{SCRATCH}/zenodo/{z}_listing.txt", z)
        for z in ["lens", "unsuccess", "group", "recenter"]}
zip_union = set().union(*zips.values())

# --- sanity: SLDE-B table counts vs paper statements
n_yy = (sldeb.model == "Y/Y").sum()
n_yn = (sldeb.model == "Y/N").sum()
n_nf = (sldeb.model == "N/-").sum()
n_nm = (sldeb.model == "NM").sum()
print(f"SLDE-B parsed: {len(sldeb)} rows (catA={len(sldeb[sldeb.cat=='A'])}, catB={len(sldeb[sldeb.cat=='B'])})")
print(f"  Y/Y={n_yy} (paper: 38 confirmed)  Y/N={n_yn} (paper: 6 ruled out)  "
      f"N/-={n_nf} (paper abstract: 9 failed)  NM={n_nm}")
assert n_yy == 38 and n_yn == 6

# --- coordinate match SLDE-B -> catalog (10 arcsec, nearest)
cat_ra = np.radians(cat.right_ascension.values)
cat_dec = np.radians(cat.declination.values)
def match_catalog(ra, dec, tol_arcsec=10.0):
    r1, d1 = np.radians(ra), np.radians(dec)
    # accurate angular separation (haversine)
    dd = cat_dec - d1
    dr = cat_ra - r1
    a = np.sin(dd / 2) ** 2 + np.cos(d1) * np.cos(cat_dec) * np.sin(dr / 2) ** 2
    sep = 2 * np.arcsin(np.sqrt(a))
    i = int(np.argmin(sep))
    sep_as = np.degrees(sep[i]) * 3600
    return (i, sep_as) if sep_as <= tol_arcsec else (None, sep_as)

sldeb["cat_idx"], sldeb["sep_as"] = zip(*[match_catalog(r.ra, r.dec) for r in sldeb.itertuples()])

records = []
used_ids = set()

SLDEA_AGG = ("SLDE-A aggregate: 488 A/B candidates modelled, 374 successful, 315 judged lenses, "
             "59 judged non-lens; per-object judgements not published")

def provenance_from_cat(row):
    p = f"Q1 catalog grade={row.grade}, expert_score={row.expert_score:.3f} ({int(row.expert_total_votes)} votes)"
    if row.subset != "discovery_engine":
        p += f", subset={row.subset}"
    return p

SPECTRO = {
    "EUCL J174907.29+645946.3": "Palomar spectroscopy: z_lens=0.481, z_source=1.839 (quality A/A; compound lens)",
    "EUCL J175049.89+665454.5": "Palomar spectroscopy: z_source=1.956 (quality A)",
    "EUCL J175555.21+635718.7": "Palomar spectroscopy: z_source=2.011 (quality A; Lyman-break galaxy)",
    "EUCL J180354.65+643421.6": "Palomar spectroscopy: z_lens=0.518, z_source=1.897 (quality A/A)",
    "EUCL J174658.82+652642.8": "Palomar spectroscopy: z_lens=0.812, z_source=2.316 (quality B/B)",
    "EUCL J174613.92+662840.2": "DESI archival spectrum: source emission line OII at z=1.303",
    "EUCL J180152.75+655455.5": ("DESI archival spectrum: second galaxy at z~0.48 vs lens z=0.36, "
                                 "supporting the non-lens verdict"),
}

# ------------------------------------------ 1) SLDE-B modeled objects (Y/Y, Y/N, N/-)
for r in sldeb[sldeb.model != "NM"].itertuples():
    matched = pd.notna(r.cat_idx)
    if matched:
        crow = cat.iloc[int(r.cat_idx)]
        name, ra, dec = crow.id_str, crow.right_ascension, crow.declination
        prov = provenance_from_cat(crow)
        used_ids.add(crow.id_str)
    else:
        name, ra, dec = r.name, r.ra, r.dec
        prov = "not in Q1 discovery-engine catalog"
    prov += f"; SLDE-B category {r.cat} (VI score {r.vi_score:g})"

    bits = []
    src = f"arXiv:2503.15325 {r.table} (cat {r.cat})"
    if r.model == "Y/Y":
        outcome = "confirmed_lens"
        bits.append("SLDE-B automated modelling (PyAutoLens SIE+shear) succeeded and experts confirmed "
                    f"lens based on the model (Model=Y/Y); theta_E={r.theta_e:g} arcsec")
        bits.append("confirmation is modelling+expert-inspection based"
                    + ("" if r.name in SPECTRO else ", not spectroscopic"))
    elif r.model == "Y/N":
        outcome = "non_lens"
        bits.append("SLDE-B automated modelling succeeded but experts determined the system is NOT a "
                    "strong lens based on the model (Model=Y/N); one of 6 grade-B systems ruled out by modelling")
    else:  # N/-
        outcome = "inconclusive"
        bits.append("SLDE-B automated modelling FAILED (Model=N/-); paper: modeller failure is not "
                    "evidence of non-lens (can fail on true lenses due to group-scale halos or "
                    "foreground contamination)")
    if r.name in SPECTRO:
        bits.append(SPECTRO[r.name])
    if matched:
        bits.append(f"SLDE-B name {r.name} (matched Q1 catalog at {r.sep_as:.2f} arcsec)")
        mem = [z for z, s in zips.items() if crow.id_str in s]
        if crow.id_str in mass.index:
            m = mass.loc[crow.id_str]
            bits.append("also in Q1 release successful-model set: theta_E_eff="
                        f"{m.einstein_radius_effective_median_pdf:.2f} arcsec (median PDF)")
            src += "; Zenodo 10.5281/zenodo.15025832 modeling_lens_mass.csv"
        elif mem:
            bits.append(f"Q1 release archive membership: {'+'.join(mem)}.zip")
            src += "; Zenodo 10.5281/zenodo.15025832"
    records.append(dict(campaign="euclid_q1_modeling", name=name, ra_deg=ra, dec_deg=dec,
                        outcome=outcome, detail="; ".join(bits),
                        candidate_provenance=prov, source_ref=src))

# ------------------------------------------ 2) SLDE-A / Zenodo release objects
cat_by_id = cat.set_index("id_str")
for i in sorted(zip_union - used_ids):
    mem = [z for z in ["lens", "group", "recenter", "unsuccess"] if i in zips[z]]
    if i in cat_by_id.index:
        crow = cat_by_id.loc[i]
        ra, dec = crow.right_ascension, crow.declination
        prov = provenance_from_cat(crow)
    else:
        ra, dec = decode_coords(i)
        prov = "not in released Q1 catalog (coordinates decoded from object_id)"
    bits = []
    if i in mass.index:
        m = mass.loc[i]
        bits.append("successfully modelled by Euclid Strong Lens Modelling Pipeline (SIE+shear): "
                    f"theta_E_eff={m.einstein_radius_effective_median_pdf:.2f} arcsec "
                    f"(1sigma {m.einstein_radius_effective_lower_1_sigma:.2f}-"
                    f"{m.einstein_radius_effective_upper_1_sigma:.2f}); in Zenodo successful-model "
                    "set (335 objects); release README: 'we hope that nearly all are strong lenses', "
                    "but per-object lens/non-lens judgement is unpublished")
        archive = "modeling_lens_mass.csv+lens.zip"
    elif mem[0] == "group":
        bits.append("group-scale lens candidate requiring multiple mass components, beyond scope of the "
                    "Q1 automated pipeline (Zenodo group.zip, 41 objects); no modelling verdict")
        archive = "group.zip"
    elif mem[0] == "recenter":
        bits.append("cutout mis-centred, which broke lens modelling (Zenodo recenter.zip, 18 objects); "
                    "no modelling verdict")
        archive = "recenter.zip"
    else:
        bits.append("modelling unsuccessful (Zenodo unsuccess.zip, 144 objects: 'modeling failures, "
                    "non-lens status, or ambiguity'); per-object reason unpublished -- the ~59 "
                    "modelling-rejected non-lenses are within this set but not individually identified")
        archive = "unsuccess.zip"
    extra = [z for z in mem if z not in archive]
    if extra:
        bits.append(f"also listed in {'+'.join(e + '.zip' for e in extra)}")
    bits.append(SLDEA_AGG)
    records.append(dict(campaign="euclid_q1_modeling", name=i, ra_deg=ra, dec_deg=dec,
                        outcome="inconclusive", detail="; ".join(bits),
                        candidate_provenance=prov,
                        source_ref=f"arXiv:2503.15324 Sect. 7.2; Zenodo 10.5281/zenodo.15025832 {archive}"))

out = pd.DataFrame.from_records(records,
        columns=["campaign", "name", "ra_deg", "dec_deg", "outcome", "detail",
                 "candidate_provenance", "source_ref"])
assert out.name.is_unique, out[out.name.duplicated(keep=False)]
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print(f"\nWrote {len(out)} rows -> {OUT_PATH}")
print("outcome counts:", out.outcome.value_counts().to_dict())
print("\nSLDE-B modeled objects matched to Q1 catalog:",
      int(sum(sldeb[sldeb.model != 'NM'].cat_idx.notna())), "of", int((sldeb.model != 'NM').sum()))
print("SLDE-B NM (not modelled, pre-Q1 data) excluded:", n_nm)
by_src = out.source_ref.str.extract(r"(arXiv:2503\.1532[45])")[0].value_counts().to_dict()
print("rows by primary paper:", by_src)
print("\nZenodo release category rows (SLDE-A, excl. SLDE-B-covered ids):")
print(out[out.source_ref.str.contains("15324")].source_ref.str.extract(r"15025832 (\S+)")[0].value_counts().to_dict())
print("\nverification vs paper totals:")
n_success = int(out.detail.str.contains("successful-model set|also in Q1 release successful-model").sum())
print(f"  successful models recovered per-object: {len(mass)} in release (paper: 374 successful; "
      "release is a later, curated snapshot)")
print(f"  SLDE-A judged-non-lens per-object: 0 recovered (paper: 59, aggregate only)")
print(f"  SLDE-B confirmed: {n_yy}/38, ruled out: {n_yn}/6, failed: {n_nf} (paper abstract says 9)")
# grade tallies of A/B coverage
ab = cat[(cat.subset == "discovery_engine") & cat.grade.isin(["A", "B"])]
print(f"  A/B candidates: {len(ab)} in catalog; {len(set(ab.id_str) & zip_union)} appear in release archives "
      f"(paper: 488 modelled)")
