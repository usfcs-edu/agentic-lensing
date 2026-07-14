#!/usr/bin/env python
"""Build euclid_q1_nisp.csv from arXiv:2604.02726v5 appendix catalogue.

Source of per-object data: v5 appendix_catalog.tex (473 rows) -- the ONLY
publicly released per-object table across all versions (v1-v3 have no
per-object table; v2, v3, v6 sources withdrawn; paper withdrawn at v6).

Outcome rules (documented for the report):
  adopted z_def = z_DESI if present else z_PHZ (paper's DESI>PHZ>none waterfall)
  - confirmed_lens: z_src MAP present AND adopted z_def present AND z_src > z_def
  - non_lens:       z_src MAP present AND adopted z_def present AND z_src <= z_def
                    (the paper's own physical-sanity check P_s = P(z_src>z_def)
                     identifies these as same-z / projected / misidentified)
  - lens_z_only:    no z_src posterior, but a SPECTROSCOPIC deflector z (DESI)
  - inconclusive:   everything else (source-z-only rows; rows whose only content
                    is a pre-existing PHZ photo-z; rows with nothing)
"""
import re
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

S = "/private/tmp/claude-501/-Users-benson-sync-research-agentic-lensing/16c65f46-2248-454d-8454-61780a59a64c/scratchpad"
APPENDIX = f"{S}/nisp_v5/appendix_catalog.tex"
LOCAL = "/Users/benson/sync/research/agentic-lensing/reproductions/euclid-q1/data/raw/q1_discovery_engine_lens_catalog.csv"
OUT = "/Users/benson/sync/research/agentic-lensing/reproductions/lensjudge/parity/data/external/euclid_q1_nisp.csv"

# ---------- parse LaTeX longtable ----------
rows = []
row_re = re.compile(r"^\s*(\d+)\s*&\s*(.+?)\s*\\\\\s*$")
with open(APPENDIX) as f:
    for line in f:
        m = row_re.match(line)
        if not m:
            continue
        num = int(m.group(1))
        fields = [x.strip() for x in m.group(2).split("&")]
        if len(fields) != 14:
            sys.exit(f"row {num}: expected 14 fields after #, got {len(fields)}: {fields}")
        ident = re.sub(r"\\texttt\{([^}]*)\}", r"\1", fields[0]).replace("\\", "").strip()

        def fnum(s):
            s = s.strip()
            if s in ("---", "--", ""):
                return np.nan
            return float(s.replace("$", "").replace("+", ""))

        ra, dec, zphz, zdesi, dzext, zdef_v6d, w68_def, zdef_grz, zsrc, w68_src, ps = [
            fnum(x) for x in fields[1:12]
        ]
        tsrc, tdef = fields[12], fields[13]
        rows.append(dict(num=num, paper_id=ident, ra=ra, dec=dec, z_phz=zphz,
                         z_desi=zdesi, zdef_v6d=zdef_v6d, w68_def=w68_def,
                         zdef_grz=zdef_grz, z_src=zsrc, w68_src=w68_src, p_s=ps,
                         t_src=tsrc, t_def=tdef))
df = pd.DataFrame(rows)
print(f"parsed rows: {len(df)} (expect 473)")
assert len(df) == 473
assert df.num.tolist() == list(range(1, 474))

# ---------- verify against v5 stated totals ----------
n_desi = df.z_desi.notna().sum()
adopted = df.z_desi.combine_first(df.z_phz)
n_adopted = adopted.notna().sum()
n_phz_adopted = (df.z_desi.isna() & df.z_phz.notna()).sum()
n_src = df.z_src.notna().sum()
tiers = df.t_src.value_counts().to_dict()
print(f"z_DESI present: {n_desi} (paper: 55)")
print(f"adopted z_def:  {n_adopted} (paper: 440) = DESI {n_desi} + PHZ {n_phz_adopted} (paper: 385)")
print(f"z_src posterior present: {n_src} (paper: 398 successful, 75 none)")
print(f"T_src tiers: {tiers} (paper: Platinum 62, Gold 230, Silver 82, Bronze 24, none 75)")

both = df.z_src.notna() & adopted.notna()
n_pass = ((df.z_src > adopted) & both).sum()
n_fail = (both & (df.z_src <= adopted)).sum()
print(f"pairs with z_src & adopted z_def: {both.sum()}; sanity pass z_src>z_def: {n_pass} "
      f"({100*n_pass/both.sum():.1f}%, paper: 97.3%); fail: {n_fail}")

# ---------- join to local SLDE catalogue by coordinates ----------
loc = pd.read_csv(LOCAL)
c_loc = SkyCoord(loc.right_ascension.values * u.deg, loc.declination.values * u.deg)
c_pap = SkyCoord(df.ra.values * u.deg, df.dec.values * u.deg)
idx, sep, _ = c_pap.match_to_catalog_sky(c_loc)
sep_arcsec = sep.arcsec
df["match_idx"] = idx
df["match_sep"] = sep_arcsec
TOL = 2.0
matched = sep_arcsec < TOL
print(f"coordinate matches < {TOL}\": {matched.sum()}/473; "
      f"sep percentiles [50,90,max of matched]: "
      f"{np.percentile(sep_arcsec[matched],50):.2f}, {np.percentile(sep_arcsec[matched],90):.2f}, "
      f"{sep_arcsec[matched].max():.2f} arcsec")
if (~matched).sum():
    print("unmatched rows (sep\"):")
    for _, r in df[~matched].iterrows():
        print(f"  #{r.num} {r.paper_id} ra={r.ra} dec={r.dec} sep={r.match_sep:.1f}")

# tile-ID consistency check for matched rows with Walmsley-style IDs
tile_ok = tile_bad = 0
for _, r in df[matched].iterrows():
    if re.fullmatch(r"\d+", r.paper_id):
        local_tile = str(loc.iloc[int(r.match_idx)].tile_index)
        if local_tile == r.paper_id:
            tile_ok += 1
        else:
            tile_bad += 1
            print(f"tile mismatch: paper #{r.num} id={r.paper_id} vs local tile "
                  f"{local_tile} ({loc.iloc[int(r.match_idx)].id_str}, sep={r.match_sep:.2f}\")")
print(f"tile-ID agreement on matched numeric-ID rows: {tile_ok} ok, {tile_bad} mismatch")

# duplicate handling: if several paper rows match the same local candidate,
# only the closest keeps the local id_str; the rest keep their paper
# designation with a provenance note (paper double-counts the system).
df["primary_match"] = False
for gidx, g in df[matched].groupby("match_idx"):
    best = g.match_sep.idxmin()
    df.loc[best, "primary_match"] = True
    if len(g) > 1:
        print(f"NOTE: paper rows {g.num.tolist()} all match local candidate "
              f"{loc.iloc[int(gidx)].id_str}; closest (row #{df.loc[best,'num']}) keeps the id_str")

# ---------- derive outcomes ----------
out_rows = []
counts = {}
for i, r in df.iterrows():
    zdef_adopted = r.z_desi if pd.notna(r.z_desi) else r.z_phz
    zdef_prov = ("DESI DR1 spec-z" if pd.notna(r.z_desi)
                 else ("Euclid PHZ MODE_1 photo-z" if pd.notna(r.z_phz) else None))
    has_src = pd.notna(r.z_src)

    if has_src and zdef_adopted is not None and pd.notna(zdef_adopted):
        outcome = "confirmed_lens" if r.z_src > zdef_adopted else "non_lens"
    elif pd.notna(r.z_desi):
        outcome = "lens_z_only"
    else:
        outcome = "inconclusive"

    d = []
    if has_src:
        d.append(f"z_src={r.z_src:.3f} (v6d posterior MAP, 68%CI_w={r.w68_src:.2f}, FoF-{r.t_src})")
    if zdef_prov:
        d.append(f"z_def={zdef_adopted:.3f} (adopted, {zdef_prov})")
    if pd.notna(r.zdef_v6d):
        d.append(f"z_def_nisp_v6d={r.zdef_v6d:.3f} (CaII-triplet posterior, 68%CI_w={r.w68_def:.2f}, FoF-{r.t_def})")
    if pd.notna(r.zdef_grz):
        d.append(f"z_def_grizli={r.zdef_grz:.3f}")
    if pd.notna(r.p_s):
        d.append(f"P(z_src>z_def)={r.p_s:.2f}")
    if outcome == "non_lens":
        d.append("fails paper physical-sanity check (z_src<=z_def MAP): same-z/projected/misID per paper Sec. FoF tiers")
    if not d:
        d.append("no spectral detection (n_det<2), no adopted z_def")
    detail = "; ".join(d)

    if matched[i]:
        lrow = loc.iloc[int(r.match_idx)]
        prov = f"grade={lrow.grade};expert_score={lrow.expert_score:.3f};votes={int(lrow.expert_total_votes)}"
        if r.primary_match:
            name = lrow.id_str
            ra_out, dec_out = float(lrow.right_ascension), float(lrow.declination)
        else:
            name = r.paper_id if r.paper_id.startswith(("Q1L-J", "DESJ")) else \
                f"{r.paper_id}_ra{r.ra:.4f}_dec{r.dec:+.4f}"
            ra_out, dec_out = r.ra, r.dec
            prov += f"; duplicate paper entry for {lrow.id_str} (paper lists this system twice)"
    else:
        name = r.paper_id if r.paper_id.startswith(("Q1L-J", "DESJ")) else \
            f"{r.paper_id}_ra{r.ra:.4f}_dec{r.dec:+.4f}"
        ra_out, dec_out = r.ra, r.dec
        prov = "no match in local SLDE catalog within 2 arcsec"

    counts[outcome] = counts.get(outcome, 0) + 1
    out_rows.append(dict(campaign="euclid_q1_nisp", name=name, ra_deg=ra_out,
                         dec_deg=dec_out, outcome=outcome, detail=detail,
                         candidate_provenance=prov,
                         source_ref="arXiv:2604.02726v5 appendix catalogue (Table: full v6d 473-row); "
                                    "single-author paper WITHDRAWN at v6"))

out = pd.DataFrame(out_rows)
assert len(out) == 473
print("\noutcome counts:", counts)
# grade distribution of matched
gm = out.candidate_provenance.str.extract(r"grade=(\w)")[0].value_counts(dropna=False)
print("grade distribution of matched rows:\n", gm)

out.to_csv(OUT, index=False)
print(f"\nwrote {len(out)} rows -> {OUT}")
