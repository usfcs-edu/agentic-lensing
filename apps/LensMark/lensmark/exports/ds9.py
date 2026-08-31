"""DS9 region export (hand-rolled writer + parser; the ``regions`` package is not a dependency).

Coordinate system ``image`` (1-based, y up - ``coords.uv_to_fits``) unless the file carries a WCS, in
which case ``fk5`` with RA/Dec offsets from the cutout centre (dE/dN arcsec -> ra = ra0 + dE/3600/cos(dec0),
dec = dec0 + dN/3600) and radii in arcsec. Every region carries ``tag={id:<item id>}`` so a file edited
in DS9 can be matched back to LensMark items.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

from .. import config, coords
from ..model import Arrow, EinsteinRing, LensMarkFile, MaskCircle, TextNote
from ..store import Campaign, atomic_write_text
from . import export_dir, exportable_items, load_files, ring_center

HEADER = "# Region file format: DS9 version 4.1"
GLOBAL = ('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 '
          'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1')
DASH = {"galaxy": "8 3", "star": "2 6", "artifact": "4 2", "einstein_ring": "1 3"}
_DS9_NAMED = {"white", "black", "red", "green", "blue", "cyan", "magenta", "yellow"}


def ds9_color(name: str) -> str:
    """LensMark palette name -> a DS9 colour: a DS9 named colour when one matches, else ``#RRGGBB``."""
    if name == "mask_red":
        return "red"
    if name == "ring_white":
        return "white"
    if name in _DS9_NAMED:
        return name
    return config.PALETTE.get(name, "#FFFFFF")


def _text(s: Optional[str]) -> str:
    return (s or "").replace("{", "(").replace("}", ")").replace("\n", " ")


class _Frame:
    """Converts uv / arcsec to the output coordinate system of one file."""

    def __init__(self, f: LensMarkFile):
        self.f = f
        self.wcs = f.image.wcs
        self.W, self.H = f.image.width, f.image.height
        self.system = "fk5" if self.wcs is not None else "image"

    def point(self, uv) -> str:
        if self.wcs is None:
            x, y = coords.uv_to_fits(uv[0], uv[1], self.W, self.H, self.f.image.array_origin)
            return f"{x:.3f},{y:.3f}"
        dE, dN = coords.uv_to_dEdN(uv[0], uv[1], self.W, self.H, self.f.image.cutout_arcsec,
                                   self.f.image.north_up, self.f.image.east_left)
        ra = self.wcs.ra_deg + dE / 3600.0 / math.cos(math.radians(self.wcs.dec_deg))
        dec = self.wcs.dec_deg + dN / 3600.0
        return f"{ra:.7f},{dec:.7f}"

    def radius(self, arcsec: float) -> str:
        if self.wcs is None:
            return f"{arcsec / self.f.image.pixel_scale_arcsec:.3f}"
        return f'{arcsec:.3f}"'


def to_ds9(file: LensMarkFile) -> str:
    fr = _Frame(file)
    lines = [HEADER,
             f"# lensmark {file.id} ({file.schema_version}) accepted|edited items; {fr.W}x{fr.H} px, "
             f"{file.image.cutout_arcsec:g} arcsec wide, {file.image.pixel_scale_arcsec:.6f} arcsec/px; "
             + ("fk5 from image.wcs" if fr.wcs else "image coords 1-based, y up"),
             GLOBAL, fr.system]
    for it in exportable_items(file):
        tag = f"tag={{id:{it.id}}}"
        if isinstance(it, MaskCircle):
            lines.append(f"circle({fr.point(it.center)},{fr.radius(it.radius_arcsec)}) # color={ds9_color(it.color)} "
                         f"dash=1 dashlist={DASH[it.kind]} width=2 text={{{_text(it.label)}}} {tag} tag={{mask}} tag={{{it.kind}}}")
        elif isinstance(it, EinsteinRing):
            label = it.label or f"theta_E ~ {it.theta_e_arcsec:.2g}\""
            lines.append(f"circle({fr.point(ring_center(file, it))},{fr.radius(it.theta_e_arcsec)}) # color={ds9_color(it.color)} "
                         f"dash=1 dashlist={DASH['einstein_ring']} width=1 text={{{_text(label)}}} {tag} tag={{einstein_ring}}")
        elif isinstance(it, Arrow):
            lines.append(f"line({fr.point(it.tail)},{fr.point(it.head)}) # line=0 1 color={ds9_color(it.color)} width=2 "
                         f"text={{{_text(it.label)}}} {tag} tag={{arrow}}")
        elif isinstance(it, TextNote):
            lines.append(f"text({fr.point(it.pos)}) # text={{{_text(it.text)}}} color={ds9_color(it.color)} {tag} tag={{text}}")
    return "\n".join(lines) + "\n"


def export_ds9(campaign: Campaign, *, out: str | Path | None = None, ids: Optional[list[str]] = None) -> list[Path]:
    paths: list[Path] = []
    for f in load_files(campaign, ids):
        if not exportable_items(f):
            continue
        path = export_dir(campaign, "ds9", out) / f"{f.id}.reg"
        atomic_write_text(path, to_ds9(f))
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- parser
_SHAPE_RE = re.compile(r"^\s*([+-]?)(circle|line|text|point|box|ellipse|polygon)\s*\(([^)]*)\)\s*(?:#?\s*(.*))?$", re.I)
_PROP_RE = re.compile(r"(\w+)\s*=\s*(\{[^}]*\}|\"[^\"]*\"|'[^']*'|[^\s]+(?:\s+-?\d+(?:\.\d+)?)*)")
_COORDSYS = {"image", "physical", "fk5", "fk4", "icrs", "galactic", "ecliptic", "j2000", "b1950"}


def _num(tok: str) -> float:
    tok = tok.strip()
    for suffix in ('"', "'", "d", "r", "p", "i"):
        if tok.endswith(suffix):
            tok = tok[:-1]
    return float(tok)


def parse_ds9(text: str) -> list[dict[str, Any]]:
    """Parse the circle/line/text/point regions of a DS9 file into dicts:
    ``{shape, coordsys, coords:[float], color, text, tags:[str], id, props:{...}}`` (``id`` from ``tag={id:...}``)."""
    out: list[dict[str, Any]] = []
    coordsys = "physical"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") and not line[1:].strip().lower().startswith(("text(", "point(")):
            continue
        if line.startswith("#"):
            line = line[1:].strip()
        if line.lower().startswith("global"):
            continue
        if line.lower() in _COORDSYS:
            coordsys = line.lower()
            continue
        m = _SHAPE_RE.match(line)
        if not m:
            continue
        sign, shape, args, props = m.group(1), m.group(2).lower(), m.group(3), m.group(4) or ""
        coords_: list[float] = []
        for tok in args.split(","):
            try:
                coords_.append(_num(tok))
            except ValueError:
                continue
        rec: dict[str, Any] = {"shape": shape, "coordsys": coordsys, "coords": coords_, "include": sign != "-",
                               "color": None, "text": None, "tags": [], "id": None, "props": {}}
        for key, val in _PROP_RE.findall(props):
            v = val.strip()
            if (v.startswith("{") and v.endswith("}")) or (v[:1] in "\"'" and v[-1:] == v[:1]):
                v = v[1:-1]
            if key == "tag":
                rec["tags"].append(v)
                if v.startswith("id:"):
                    rec["id"] = v[3:]
            elif key == "text":
                rec["text"] = v
            elif key == "color":
                rec["color"] = v
            else:
                rec["props"][key] = v
        out.append(rec)
    return out
