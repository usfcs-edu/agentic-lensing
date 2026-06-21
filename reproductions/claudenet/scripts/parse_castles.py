#!/usr/bin/env python
"""Parse CASTLES noimages.html into normalized confirmed-lens rows.

Output columns EXACTLY: name, ra (deg float), dec (deg float), grade, source, ref
source tag = "CASTLES_SLED". Keep only dec <= +32.375 (DR11-south footprint).
"""
import re
import sys
import math
import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u

HTML = "/home2/benson/git/agentic-lensing/reproductions/claudenet/data/harvest/castles_noimages.html"
OUT  = "/home2/benson/git/agentic-lensing/reproductions/claudenet/data/harvest/castles_sled.parquet"
DEC_CUT = 32.375

REF = ("CASTLES (CfA-Arizona Space Telescope LEns Survey), "
       "Kochanek, Falco, Impey, Lehar, McLeod, Rix; "
       "https://lweb.cfa.harvard.edu/castles/")

# sexagesimal patterns.
# RA: hh:mm:ss(.s)  (no sign)
RA_RE  = re.compile(r"^\s*\d{1,2}:\d{1,2}:\d{1,2}(\.\d+)?\s*$")
# Dec: optional sign, dd:mm:ss(.s). Source has two known typos handled below:
#   - sign sometimes omitted (H1413+117 -> "11:29:41.4")
#   - one row uses a '.' instead of ':' as a field separator (Q0252-3249 -> "-32:49.10.1")
# So we accept ':' or '.' as separators and an optional leading sign.
DEC_RE = re.compile(r"^\s*[+-]?\d{1,2}[:.]\d{1,2}[:.]\d{1,2}(\.\d+)?\s*$")
GRADE_RE = re.compile(r"^[ABC]$")

def normalize_dec(s):
    """Repair the two known CASTLES dec-string typos, return a clean +dd:mm:ss.s string."""
    s = s.strip()
    sign = ""
    if s[0] in "+-":
        sign, body = s[0], s[1:]
    else:
        sign, body = "+", s
    # split on either ':' or '.', but keep a trailing fractional second.
    # Expected 3 main fields (deg, min, sec[.frac]).
    parts = re.split(r"[:.]", body)
    # parts like ['32','49','10','1'] for "-32:49.10.1" -> deg,min,sec,frac
    if len(parts) == 4:
        d, m, sec, frac = parts
        sec = f"{sec}.{frac}"
    elif len(parts) == 3:
        d, m, sec = parts
    else:
        # unexpected; let astropy try the raw string
        return s
    return f"{sign}{int(d):02d}:{int(m):02d}:{sec}"

def strip_tags(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&nbsp", " ")
    s = s.replace("&plusmn;", "+/-")
    return s.strip()

def main():
    with open(HTML, encoding="latin-1") as f:
        html = f.read()

    # split into <tr> ... </tr> blocks (case-insensitive)
    rows = re.split(r"(?i)<tr[^>]*>", html)
    parsed = []
    skipped_no_coords = 0
    for block in rows:
        if "Individual/" not in block:
            continue
        # cut at </tr> if present
        block = re.split(r"(?i)</tr>", block)[0]
        # extract name from the Individual/ anchor text
        m = re.search(r"(?i)<a\s+href=[\"']?Individual/[^>]*>(.*?)</a>", block)
        if not m:
            continue
        name = strip_tags(m.group(1))
        if not name:
            continue
        # all <td> cell contents
        cells_raw = re.findall(r"(?i)<td[^>]*>(.*?)(?=<td|</tr>|$)", block, flags=re.S)
        cells = [strip_tags(c) for c in cells_raw]
        # find RA and Dec cells by pattern
        ra_str = dec_str = None
        for i, c in enumerate(cells):
            if ra_str is None and RA_RE.match(c):
                # ensure the following dec-like cell exists nearby
                ra_str = c.strip()
                ra_idx = i
                continue
            if ra_str is not None and dec_str is None and DEC_RE.match(c):
                dec_str = c.strip()
                break
        # grade: first standalone A/B/C cell that appears before RA
        grade = "?"
        for c in cells:
            cc = c.strip()
            if GRADE_RE.match(cc):
                grade = cc
                break
        if ra_str is None or dec_str is None:
            skipped_no_coords += 1
            continue
        try:
            ra_deg = Angle(ra_str, unit=u.hourangle).to(u.deg).value
            dec_deg = Angle(normalize_dec(dec_str), unit=u.deg).value
        except Exception as e:
            print(f"  parse fail {name}: ra={ra_str!r} dec={dec_str!r} -> {e}", file=sys.stderr)
            skipped_no_coords += 1
            continue
        parsed.append(dict(name=name, ra=ra_deg, dec=dec_deg, grade=grade,
                           source="CASTLES_SLED", ref=REF))

    df = pd.DataFrame(parsed, columns=["name", "ra", "dec", "grade", "source", "ref"])
    print(f"rows parsed from CASTLES HTML (with coords): {len(df)}")
    print(f"rows skipped (no/invalid coords): {skipped_no_coords}")

    # drop null / non-finite ra/dec
    before = len(df)
    df = df[df["ra"].apply(lambda x: x is not None and math.isfinite(x))]
    df = df[df["dec"].apply(lambda x: x is not None and math.isfinite(x))]
    df = df.reset_index(drop=True)
    if len(df) != before:
        print(f"dropped {before - len(df)} rows with null/non-finite ra/dec")

    # enforce dtypes
    df["name"] = df["name"].astype(str)
    df["ra"] = df["ra"].astype(float)
    df["dec"] = df["dec"].astype(float)
    df["grade"] = df["grade"].astype(str)
    df["source"] = df["source"].astype(str)
    df["ref"] = df["ref"].astype(str)

    print(f"ROW COUNT BEFORE dec cut: {len(df)}")
    print("grade breakdown (all, before dec cut):")
    print(df["grade"].value_counts().to_dict())

    df_south = df[df["dec"] <= DEC_CUT].reset_index(drop=True)
    print(f"ROW COUNT AFTER dec cut (dec <= {DEC_CUT}): {len(df_south)}")
    print("grade breakdown (south):")
    print(df_south["grade"].value_counts().to_dict())

    df_south.to_parquet(OUT, index=False)
    print(f"WROTE: {OUT}")
    # sanity readback
    rb = pd.read_parquet(OUT)
    print("readback shape:", rb.shape, "| dtypes:", dict(rb.dtypes.astype(str)))
    print("dec range:", float(rb["dec"].min()), "->", float(rb["dec"].max()))
    print("ra range:", float(rb["ra"].min()), "->", float(rb["ra"].max()))
    print("\nsample rows:")
    print(rb.head(5).to_string())

if __name__ == "__main__":
    main()
