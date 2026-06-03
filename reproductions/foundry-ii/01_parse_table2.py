#!/usr/bin/env python
"""Foundry II (Huang et al. 2025b) - parse Table 2 (the 73-system DESI EDR table).

Table 2 columns (from PDF caption):
 (1) System Name      - "DESI J<RA>+<Dec>" (confirmed) or "DESI-<RA><Dec>" (candidate)
                        RA/Dec are DECIMAL DEGREES with 4 dp (not sexagesimal).
 (2) Discovery Paper  - citation legend number(s)
 (3) Discovery Name   - HSC/SDSS/PS1 name (often absent)
 (4) z_d              - lens (deflector) redshift
 (5) z_s              - source redshift (often absent / not measured)
 (6) sigma_v [km/s]   - velocity dispersion "val +/- err" (FastSpecFit), or "-"
 (7) VI               - which redshift was corrected/confirmed by visual inspection

The text is extracted page-by-page with pdfplumber (per-page extract_text gives
clean rows; whole-doc extraction scrambles columns).

We parse each data row, pull RA/Dec straight from the name, and bucket each
system by its section header (confirmed / known / pending-source / pending-zs /
pending-both / nonlens).
"""
import re, csv, os

PG4 = open(os.path.join(os.path.dirname(__file__), "data/table2_pg4.txt")).read()
PG5 = open(os.path.join(os.path.dirname(__file__), "data/table2_pg5.txt")).read()

SECTION_PATTERNS = [
    (re.compile(r"Confirmed systems"), "confirmed"),
    (re.compile(r"Known system with lens observed"), "known"),
    (re.compile(r"Confirmation pending with source not yet observed"), "pending_source"),
    (re.compile(r"Confirmation pending with z\b.*not confirmed \(13\)|Confirmation pending with z.*not confirmed.*13"), "pending_zs"),
    (re.compile(r"Confirmation pending with z and z .*not confirmed \(1\)|Confirmation pending with z and z"), "pending_both"),
    (re.compile(r"Confirmed Nonlenses"), "nonlens"),
]

# A name token: "DESI J149.8209+01.0331" or "DESI-214.8006+53.4366"
# RA always 3 digits.ddddd ; Dec sign + 2 digits.dddd ; optional trailing letter (b/c/d)
NAME_RE = re.compile(r"^(DESI[ ]?J?[-]?)(\d{2,3}\.\d{3,4})([+-]\d{2}\.\d{3,4})([a-z]?)\b")
REDSHIFT_RE = re.compile(r"^\d\.\d{3,4}$")
SIGMA_RE = re.compile(r"(\d{2,4})\s*[±+]/?-?\s*(\d{1,4})")  # val +/- err
SIGMA_RE2 = re.compile(r"(\d{2,4})±(\d{1,4})")


def classify_section(line):
    for pat, name in SECTION_PATTERNS:
        if pat.search(line):
            return name
    return None


def parse_name(tok):
    """Return (canonical_name, ra_deg, dec_deg, suffix) or None."""
    m = NAME_RE.match(tok)
    if not m:
        return None
    ra = float(m.group(2))
    dec = float(m.group(3))
    suffix = m.group(4)
    return m.group(0), ra, dec, suffix


def parse_rows(text, start_section=None):
    """Yield (section, raw_line, parsed_dict). start_section carries the
    open section header across a page break."""
    section = start_section
    rows = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        sec = classify_section(line)
        if sec:
            section = sec
            continue
        if section is None:
            continue
        # is this a data row? must start with a DESI name token
        nm = parse_name(line)
        if nm is None:
            # could be a continuation line (e.g. second source redshift "1.4794 z s")
            rows.append((section, line, None))
            continue
        name, ra, dec, suffix = nm
        rest = line[len(name):].strip()
        rows.append((section, line, {"name": name, "ra": ra, "dec": dec,
                                     "suffix": suffix, "rest": rest}))
    return rows


def extract_redshifts_and_sigma(rest):
    """From the part after the name, pull (z_d, z_s, sigma_v, sigma_err).

    Row layout after name:  <paper_cite(s)> [<disc_name>] <z_d> [<z_s>] <sig> ± <err> [VI]
    Normalise the +/- glyph, split on the sigma anchor: redshift floats are the
    float tokens BEFORE the sigma block; sigma val/err straddle the +/- glyph.
    A trailing 'c' on a redshift (e.g. 2.2066c) flags a non-DESI (Keck) source z.
    """
    # normalise the plus-minus glyph and squash "val ± err" spacing variants
    s = rest.replace("±", " ± ")
    s = re.sub(r"\s+", " ", s).strip()

    sig = sigerr = None
    z_d = z_s = None

    m = re.search(r"(\d{2,4})\s*±\s*(\d{1,4})", s)
    if m:
        sig = int(m.group(1)); sigerr = int(m.group(2))
        before = s[:m.start()]
    else:
        before = s  # sigma may be absent (e.g. "-")

    # redshift floats are X.XXX or X.XXXX tokens (optionally suffixed 'c').
    # HSC/SDSS/PS1 discovery names contain digits but NOT a bare 'D.DDDD' float
    # preceded by a space and not part of a longer token, so this is safe.
    ztoks = re.findall(r"(?<![\w.])(\d\.\d{3,4})c?(?![\w])", before)
    zvals = [float(z) for z in ztoks]
    if len(zvals) >= 1:
        z_d = zvals[0]
    if len(zvals) >= 2:
        z_s = zvals[1]
    return z_d, z_s, sig, sigerr


def main():
    rows4 = parse_rows(PG4)
    # carry the last open section header from page 4 into page 5
    last_sec = next((sec for sec, _, _ in reversed(rows4) if sec), None)
    rows5 = parse_rows(PG5, start_section=last_sec)
    all_rows = rows4 + rows5
    systems = []
    pending_continuation = None
    for section, line, parsed in all_rows:
        if parsed is None:
            # continuation line: e.g. "1.4794 z s" -> second source for prior system
            if systems and re.match(r"^\d\.\d{3,4}", line):
                z2 = float(re.match(r"^(\d\.\d{3,4})", line).group(1))
                systems[-1]["z_s2"] = z2
            continue
        z_d, z_s, sig, sigerr = extract_redshifts_and_sigma(parsed["rest"])
        # discovery name (HSC/SDSS/PS1) if present
        dm = re.search(r"((?:HSC|SDSS|PS1|HSCJ)\S*\d[\w+\-]*)", parsed["rest"])
        disc = dm.group(1) if dm else ""
        systems.append({
            "name": parsed["name"],
            "section": section,
            "ra_deg": parsed["ra"],
            "dec_deg": parsed["dec"],
            "suffix": parsed["suffix"],
            "z_lens_pub": z_d,
            "z_source_pub": z_s,
            "z_s2": None,
            "sigma_v_pub": sig,
            "sigma_v_err_pub": sigerr,
            "discovery_name": disc,
            "raw": parsed["rest"],
        })
    return systems


if __name__ == "__main__":
    systems = main()
    out = os.path.join(os.path.dirname(__file__), "data/foundry_ii_table2.csv")
    cols = ["name", "section", "ra_deg", "dec_deg", "suffix", "z_lens_pub",
            "z_source_pub", "z_s2", "sigma_v_pub", "sigma_v_err_pub",
            "discovery_name", "raw"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in systems:
            w.writerow(s)
    # summary
    from collections import Counter
    c = Counter(s["section"] for s in systems)
    print(f"Parsed {len(systems)} systems")
    for k, v in c.items():
        print(f"  {k}: {v}")
    nzd = sum(1 for s in systems if s["z_lens_pub"] is not None)
    nzs = sum(1 for s in systems if s["z_source_pub"] is not None)
    nsig = sum(1 for s in systems if s["sigma_v_pub"] is not None)
    print(f"  z_lens present: {nzd}; z_source present: {nzs}; sigma_v present: {nsig}")
    print(f"Wrote {out}")
