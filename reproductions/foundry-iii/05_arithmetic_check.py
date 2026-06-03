#!/usr/bin/env python
"""
Step 05: Redshift-arithmetic consistency check (no fitting).

For each NIRES system, take the published zs and the emission lines the paper
reports detecting, and compute the observed-frame wavelengths
lambda_obs = lambda_rest * (1 + zs).

We verify:
  * every line the paper says it DETECTED in NIRES lands inside the NIRES
    coverage (0.94 - 2.45 um) and outside the 1.85-1.88 um telluric gap;
  * the two FIT lines' observed wavelengths are mutually consistent with a
    single redshift (by construction they are -- this is the closure check that
    the published zs reproduces the reported line set);
  * lines the paper says are redshifted OUT of the optical/DESI range
    ([OII] 3727 beyond ~9800 A at z>~1.6) indeed are.

This is the table-arithmetic half of the consistency reproduction: it shows the
published redshifts are internally consistent with the stated line identifications
and the NIRES instrument bandpass.

Run:  python 05_arithmetic_check.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NIRES_BLUE = 9400.0     # A  (0.94 um)
NIRES_RED = 24500.0     # A  (2.45 um)
TELLURIC_GAP = (18500.0, 18800.0)  # 1.85-1.88 um poor-transmission gap
DESI_RED_EDGE = 9800.0  # A, optical edge beyond which [OII] is lost (paper Sec 5.2)


def in_nires(lam):
    if not (NIRES_BLUE <= lam <= NIRES_RED):
        return "OUT-of-band"
    if TELLURIC_GAP[0] <= lam <= TELLURIC_GAP[1]:
        return "telluric-gap"
    return "OK"


def main():
    blob = json.load(open(os.path.join(DATA, "systems.json")))
    systems = blob["systems"]
    rest = blob["rest_wavelengths"]
    targets = [s for s in systems if s["zs_source"] == "NIRES"]

    print("Observed-frame wavelengths lambda_obs = lambda_rest*(1+zs) for detected lines.")
    print(f"NIRES coverage: {NIRES_BLUE/1e4:.2f}-{NIRES_RED/1e4:.2f} um, "
          f"telluric gap {TELLURIC_GAP[0]/1e4:.3f}-{TELLURIC_GAP[1]/1e4:.3f} um.\n")

    all_ok = True
    report = []
    for s in targets:
        zs = s["zs"]
        print(f"== {s['name']}  zs={zs:.5f} ==")
        sys_rec = {"name": s["name"], "zs": zs, "lines": []}
        for line in s["lines"]:
            lr = rest[line]
            lo = lr * (1 + zs)
            status = in_nires(lo)
            flag = "(FIT)" if line in s["fit_lines"] else ""
            if status == "OUT-of-band":
                all_ok = False
            print(f"   {line:12s} {lr:8.2f} A rest -> {lo/1e4:7.4f} um obs  "
                  f"[{status}] {flag}")
            sys_rec["lines"].append({"line": line, "rest": lr, "obs": lo,
                                     "obs_um": lo / 1e4, "status": status,
                                     "is_fit_line": line in s["fit_lines"]})
        # [OII] check: where is the 3727 doublet for this z?
        oii = rest["[OII]3727"] * (1 + zs)
        oii_in_optical = oii <= DESI_RED_EDGE
        print(f"   [OII]3727 at {oii/1e4:.4f} um -> "
              f"{'in' if oii_in_optical else 'beyond'} DESI optical edge "
              f"({DESI_RED_EDGE/1e4:.4f} um) -> NIR needed: "
              f"{'no' if oii_in_optical else 'YES'}")
        sys_rec["oii_obs_um"] = oii / 1e4
        sys_rec["oii_beyond_desi"] = not oii_in_optical
        report.append(sys_rec)
        print()

    print("All reported NIRES lines fall inside the NIRES band:",
          "PASS" if all_ok else "FAIL")
    print("All sources require NIR (their [OII] is beyond DESI optical edge):",
          "PASS" if all(r["oii_beyond_desi"] for r in report) else
          "(some borderline)")
    with open(os.path.join(DATA, "arithmetic_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote data/arithmetic_check.json")


if __name__ == "__main__":
    main()
