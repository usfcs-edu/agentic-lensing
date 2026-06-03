"""01 - Build the ground-truth catalog of confirmed lens systems from Lin et al. 2025
(DESI Strong Lens Foundry IV, arXiv:2509.18087, Table 2 + Section 4.1).

The paper's Table 2 is column-shredded by any PDF text extractor (object / RA / Dec /
z / Qz live in separate vertical blocks), so a direct table parse is brittle. Instead
we parse the per-system narrative in Section 4.1, which states each system's name,
the lens redshift(s), and the source redshift(s) in plain prose ("z = 0.431",
"source redshift of z = 0.908", ...). We hand-curate the first ~13 systems (the ones
with the cleanest prose and the ones we have public MUSE cubes for) into a CSV that
later scripts use as the comparison ground truth.

This is honest: the redshifts here are transcribed from the published paper, NOT
measured by us. Scripts 02-05 do the *measurement* from public ESO MUSE cubes and
compare back to this file.

Output: data/confirmed_catalog.csv
"""
from pathlib import Path
import csv

REPRO = Path(__file__).parent
DATA = REPRO / "data"
DATA.mkdir(exist_ok=True)

# (name, RA_deg, Dec_deg, z_lens, z_source, source_lines, lens_lines, prog_id, muse_target, Qz_src)
# Coordinates are the LENS (L1) coords from Table 2 (deg). z's transcribed from Sect 4.1.
# muse_target = the generic "LensNN" name under which the public MUSE cube is archived
# (recovered by coordinate cross-match in script 02).
CONFIRMED = [
    # name                      RA        Dec       z_lens  z_src   src_features                 lens_features                prog            muse     Qz
    ("DESI J003.6745-13.5042",   3.6745, -13.5042, 0.431,  0.908,  "[OII]3727,[OIII]4364",       "CaHK,Gband",                "109.238W.004", "Lens16",  1),
    ("DESI J043.6663-04.3068",  43.6663,  -4.3068, 0.345,  2.45,   "SiIV,CIV,FeII,[CIII]1909",   "CaHK,Gband",                "111.24P8.001", "Lens16",  1),
    ("DESI J053.6251-13.1869",  53.6251, -13.1869, 0.387,  2.3,    "SiII,CIV,FeII,[CIII]1909",   "CaHK,Gband",                "111.24P8.001", "Lens19",  1),
    ("DESI J055.0894-25.5581",  55.0894, -25.5581, 0.656,  2.682,  "[OII]+SiII1303,CII,[CIII]",  "CaHK,Gband",                "111.24P8.001", "Lens20",  2),
    ("DESI J060.5238-22.0990",  60.5238, -22.0990, 0.467,  0.821,  "[OII]3727,Hgamma,[OIII]5007","CaHK,Gband",                "111.24P8.001", "Lens22",  1),
    ("DESI J065.6453-28.0646",  65.6453, -28.0646, 0.62,   1.175,  "CaHK,[OII]3727",             "CaHK,Gband",                "111.24P8.001", "Lens24",  1),
    ("DESI J073.5286-10.2227",  73.5286, -10.2227, 0.248,  1.0436, "[OII]3727,CaHK",             "CaHK,Gband",                "111.24P8.001", "Lens26",  1),
    ("DESI J073.9027-25.5132",  73.9027, -25.5132, 0.378,  2.82,   "SiIV,CIV,FeII,[CIII]1909",   "[OII],Hbeta,[OIII],Halpha", "111.24P8.001", "Lens27",  1),
    ("DESI J074.9646-30.7233",  74.9646, -30.7233, 0.441,  1.4488, "[OII]3727",                  "CaHK,Gband",                "111.24P8.001", "Lens28",  1),
    ("DESI J075.2793-24.4176",  75.2793, -24.4176, 0.32,   2.83,   "SiII,CIV,FeII,[CIII]1909",   "CaHK,Gband",                "111.24P8.001", "Lens29",  1),
    ("DESI J086.3072-26.5878",  86.3072, -26.5878, 0.275,  2.173,  "SiII,CIV,FeII,[CIII]1909",   "CaHK,Gband",                "111.24UJ.008", "Lens5",   2),
    ("DESI J087.1525-36.2427",  87.1525, -36.2427, 0.301,  0.8453, "[OII]3727,Hbeta,[OIII]5007", "CaHK,Gband",                "111.24P8.001", "Lens32",  1),
    ("DESI J090.9854-35.9683",  90.9854, -35.9683, 0.489,  1.432,  "[OII]3727",                  "CaHK,Gband",                "111.24UJ.008", "Lens1DifferentNight2", 1),
]

HEADER = ["name", "ra_deg", "dec_deg", "z_lens", "z_source",
          "source_features", "lens_features", "prog_id", "muse_target", "qz_source"]

def main():
    out = DATA / "confirmed_catalog.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for row in CONFIRMED:
            w.writerow(row)
    print(f"Wrote {len(CONFIRMED)} confirmed systems -> {out}")
    print(f"  z_lens   range: {min(r[3] for r in CONFIRMED):.3f} - {max(r[3] for r in CONFIRMED):.3f}")
    print(f"  z_source range: {min(r[4] for r in CONFIRMED):.3f} - {max(r[4] for r in CONFIRMED):.3f}")
    # quick MUSE-observability sanity (MUSE covers 4750-9350 A)
    print("\nObservability of key features in the MUSE window (4750-9350 A):")
    for name, ra, dec, zl, zs, *_ in CONFIRMED[:6]:
        cak = 3934.0 * (1 + zl)          # Ca K (lens)
        oii = 3727.0 * (1 + zs)          # [OII] (source)
        print(f"  {name:26}  lens CaK obs={cak:6.0f}A  src [OII] obs={oii:6.0f}A")

if __name__ == "__main__":
    main()
