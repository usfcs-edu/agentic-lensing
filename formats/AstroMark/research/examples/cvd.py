#!/usr/bin/env python3
"""Colour-vision-deficiency simulation and colour-difference metrics for AstroMark.

    python cvd.py simulate IN.png OUT_DIR          # writes deuter/prot/trit/grey variants
    python cvd.py pairtest PALETTE.json            # the redundancy gate, as a pass/fail test
    python cvd.py selftest

Why this exists as its own module: the project's accessibility requirement is that a notation must
be fully legible without colour, with colour as redundant reinforcement only (WCAG 2.2 SC 1.4.1,
"Use of Color"). That is normally checked by a human squinting at a simulation. Here it is a test.

**The pair test — the actual gate.** For every pair of semantically distinct marks that can appear
on the same panel, compute CIEDE2000 between their inks under normal vision and under each CVD
simulation. The criterion is deliberately NOT "colour difference is large". It is:

    for every pair whose dE2000 < DELTA_E_FLOOR under ANY simulation,
    assert that glyph shape OR stroke texture OR orientation differs.

That is the WCAG rule written as an assertion: colour may reinforce a distinction, but something
non-colour must also carry it. A palette passes not by being colourful but by never being the ONLY
thing separating two meanings.

Simulation uses Machado, Oliveira & Fernandes (2009), "A Physiologically-based Model for Simulation
of Color Vision Deficiency", IEEE TVCG 15(6). The severity-1.0 matrices are inlined below rather
than imported from a library, deliberately: a library that changes version would make the
verification itself non-reproducible.

Matrices operate on LINEAR RGB, so sRGB is decoded before and re-encoded after. Skipping that step
is the single most common error in CVD simulation code and it materially changes the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

# Machado, Oliveira & Fernandes (2009), Table 1, severity 1.0. Linear-RGB operators.
MACHADO_1_0: dict[str, np.ndarray] = {
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritanopia": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}

# Prevalence, northern-European ancestry, for the report's framing (Birch 2012).
PREVALENCE = {
    "deuteranomaly": "~4.6% of males",
    "protanomaly": "~1.1% of males",
    "deuteranopia": "~1.3% of males",
    "protanopia": "~1.0% of males",
    "tritanopia": "<0.01%, sex-independent",
    "all_male": "~8% of males, ~0.5% of females",
}

# Floor for "these two inks are not reliably distinguishable in a thin stroke or small glyph".
# Calibrated, not guessed: pure red #FF0000 against pure green #00E000 — the canonical confusion —
# still measures dE2000 = 12.9 under deuteranopia simulation, because almost all of what survives is
# a LIGHTNESS difference (L* 53 vs 78), not hue. A floor at 12 would therefore wave through the exact
# pair the criterion exists to catch. 15 is set above that residual. Large filled areas tolerate a
# lower floor than the thin strokes and small terminators this notation is made of.
DELTA_E_FLOOR = 15.0


# --- sRGB <-> linear -------------------------------------------------------------------------

def srgb_to_linear(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(a: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / 2.4) - 0.055)


def simulate(rgb01: np.ndarray, kind: str) -> np.ndarray:
    """rgb01: (..., 3) in sRGB 0..1. Returns the same shape, simulated."""
    if kind == "greyscale":
        lin = srgb_to_linear(rgb01)
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        return linear_to_srgb(np.stack([y, y, y], axis=-1))
    if kind not in MACHADO_1_0:
        raise ValueError(f"unknown deficiency {kind!r}; "
                         f"choose from {sorted(MACHADO_1_0) + ['greyscale']}")
    lin = srgb_to_linear(rgb01)
    out = lin @ MACHADO_1_0[kind].T
    return linear_to_srgb(out)


def simulate_image(src: Path, kind: str) -> Image.Image:
    im = Image.open(src).convert("RGB")
    a = np.asarray(im, dtype=np.float64) / 255.0
    out = simulate(a, kind)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB")


# --- CIE Lab and CIEDE2000 -------------------------------------------------------------------

D65 = np.array([0.95047, 1.00000, 1.08883])
M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    xyz = srgb_to_linear(rgb01) @ M_RGB2XYZ.T / D65
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray, kL=1.0, kC=1.0, kH=1.0) -> float:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005 formulation)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7))) if Cbar > 0 else 0.5
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0 if (a1p or b1) else 0.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0 if (a2p or b2) else 0.0

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        dhp = dh - 360.0 if dh > 180.0 else (dh + 360.0 if dh < -180.0 else dh)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    if C1p * C2p == 0:
        hbp = h1p + h2p
    else:
        s = h1p + h2p
        if abs(h1p - h2p) > 180.0:
            hbp = (s + 360.0) / 2.0 if s < 360.0 else (s - 360.0) / 2.0
        else:
            hbp = s / 2.0

    T = (1.0 - 0.17 * np.cos(np.radians(hbp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))
    dTheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7)) if Cbp > 0 else 0.0
    Sl = 1.0 + (0.015 * (Lbp - 50.0) ** 2) / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2.0 * dTheta)) * Rc

    return float(np.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                         + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))))


def hex_to_rgb01(s: str) -> np.ndarray:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def delta_e_under(hex_a: str, hex_b: str, kind: str | None) -> float:
    a, b = hex_to_rgb01(hex_a), hex_to_rgb01(hex_b)
    if kind:
        a, b = simulate(a, kind), simulate(b, kind)
    return ciede2000(rgb_to_lab(a), rgb_to_lab(b))


# --- the redundancy gate ----------------------------------------------------------------------

CONDITIONS = [None, "deuteranopia", "protanopia", "tritanopia", "greyscale"]


def pair_test(marks: list[dict], floor: float = DELTA_E_FLOOR) -> dict:
    """The accessibility gate.

    `marks` is a list of {name, color, shape, texture, orientation, co_occurs}. Two marks that can
    appear on the same panel and are closer than `floor` in CIEDE2000 under any condition MUST
    differ in shape, texture or orientation. Returns a report; `ok` is the gate.
    """
    failures, checked, near = [], 0, 0
    for m in marks:
        for key in ("name", "color", "shape", "texture"):
            if key not in m:
                raise ValueError(f"mark {m.get('name', '?')} is missing {key!r}")
    for a, b in combinations(marks, 2):
        # marks that can never share a panel cannot be confused on one
        if not (set(a.get("co_occurs", ["*"])) & set(b.get("co_occurs", ["*"])) or
                "*" in a.get("co_occurs", ["*"]) or "*" in b.get("co_occurs", ["*"])):
            continue
        checked += 1
        worst_kind, worst = None, float("inf")
        for kind in CONDITIONS:
            d = delta_e_under(a["color"], b["color"], kind)
            if d < worst:
                worst, worst_kind = d, kind or "normal"
        if worst >= floor:
            continue
        near += 1
        redundant = (a["shape"] != b["shape"]
                     or a["texture"] != b["texture"]
                     or a.get("orientation") != b.get("orientation"))
        if not redundant:
            failures.append({
                "pair": [a["name"], b["name"]],
                "delta_e": round(worst, 2),
                "worst_condition": worst_kind,
                "shape": [a["shape"], b["shape"]],
                "texture": [a["texture"], b["texture"]],
                "reason": "indistinguishable by colour and identical in every non-colour channel",
            })
    return {
        "ok": not failures,
        "pairs_checked": checked,
        "pairs_below_floor": near,
        "floor_delta_e": floor,
        "conditions": [c or "normal" for c in CONDITIONS],
        "failures": failures,
        "note": ("A pair below the floor is not itself a failure — it is a failure only when no "
                 "non-colour channel separates the two. That is WCAG 2.2 SC 1.4.1 as an assertion."),
    }


def selftest() -> int:
    ok = True

    def check(label, got, want, tol=None):
        nonlocal ok
        good = abs(got - want) <= tol if tol is not None else got == want
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {label}: {got!r}" + (f" (want {want!r})" if not good else ""))

    print("sRGB round trip")
    v = np.linspace(0, 1, 11)
    check("linear->srgb->linear max err", float(np.abs(srgb_to_linear(linear_to_srgb(v)) - v).max()), 0.0, 1e-9)

    print("CIEDE2000 against Sharma, Wu & Dalal (2005) reference pairs")
    # Sharma's published test data — the pairs that break naive implementations.
    for lab1, lab2, want in [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
        ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ]:
        check(f"dE {lab1[0]:.0f}/{lab2[0]:.0f}", round(ciede2000(np.array(lab1), np.array(lab2)), 4), want, 1e-3)

    print("CVD simulation sanity")
    red, green = hex_to_rgb01("#FF0000"), hex_to_rgb01("#00E000")
    d_norm = ciede2000(rgb_to_lab(red), rgb_to_lab(green))
    d_deut = ciede2000(rgb_to_lab(simulate(red, "deuteranopia")),
                       rgb_to_lab(simulate(green, "deuteranopia")))
    print(f"       red vs green: normal dE={d_norm:.1f}, deuteranopia dE={d_deut:.1f}")
    check("deuteranopia collapses red/green", d_deut < d_norm * 0.5, True)
    blue, yellow = hex_to_rgb01("#0060FF"), hex_to_rgb01("#E8E000")
    check("deuteranopia preserves blue/yellow",
          ciede2000(rgb_to_lab(simulate(blue, "deuteranopia")),
                    rgb_to_lab(simulate(yellow, "deuteranopia"))) > 40.0, True)
    check("greyscale is achromatic",
          float(np.abs(np.diff(simulate(hex_to_rgb01("#FF6B6B"), "greyscale"))).max()), 0.0, 1e-9)

    print("pair test gate")
    r = pair_test([
        {"name": "a", "color": "#FF0000", "shape": "tri", "texture": "solid"},
        {"name": "b", "color": "#00E000", "shape": "tri", "texture": "solid"},
    ])
    check("red/green identical glyphs fails the gate", r["ok"], False)
    r2 = pair_test([
        {"name": "a", "color": "#FF0000", "shape": "tri", "texture": "solid"},
        {"name": "b", "color": "#00E000", "shape": "chevron", "texture": "dash"},
    ])
    check("same colours, different glyph+texture passes", r2["ok"], True)

    print("\n" + ("selftest PASSED" if ok else "selftest FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="write CVD variants of an image")
    s.add_argument("src", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--kinds", nargs="+",
                   default=["deuteranopia", "protanopia", "tritanopia", "greyscale"])
    p = sub.add_parser("pairtest", help="run the redundancy gate over a mark table")
    p.add_argument("marks_json", type=Path, help='JSON list of {name,color,shape,texture[,orientation,co_occurs]}')
    p.add_argument("--floor", type=float, default=DELTA_E_FLOOR)
    sub.add_parser("selftest")
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "simulate":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.src.stem
        for kind in args.kinds:
            dest = args.out_dir / f"{stem}-{kind[:5]}.png"
            simulate_image(args.src, kind).save(dest, optimize=False)
            print(f"wrote {dest}")
        return 0

    marks = json.loads(args.marks_json.read_text(encoding="utf-8"))
    if isinstance(marks, dict):
        marks = marks.get("marks", [])
    rep = pair_test(marks, args.floor)
    print(json.dumps(rep, indent=2))
    if not rep["ok"]:
        print(f"\nGATE FAILED: {len(rep['failures'])} pair(s) separated by colour alone.",
              file=sys.stderr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
