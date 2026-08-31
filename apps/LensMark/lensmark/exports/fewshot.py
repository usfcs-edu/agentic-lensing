"""Few-shot bundle (``exports/fewshot/``): K finished images as (original, annotated PNG, LensMark JSON,
markdown card) tuples plus a content-addressed manifest. ``prompt_sha256`` is the sha256 of the JSON and
markdown texts concatenated in manifest order, so the propose pipeline can record exactly which examples
a run saw (``ProposalRun.fewshot_sha256``). No embargo logic - this corpus is not blind (Greg, 2026-08-30)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..critique import latest_critique
from ..model import Arrow, EinsteinRing, LensMarkFile, MaskCircle, TextNote
from ..store import Campaign, atomic_write_text
from . import EXPORT_STATUSES, export_dir, exportable_items, load_files, ring_center

SCHEMA_VERSION = "lensmark-fewshot/1.0"
_BUNDLE_FILE_RE = re.compile(r"^\d{3}-.*\.(png|json|md)$")


def eligible(campaign: Campaign, file: LensMarkFile, *, require_flag: bool) -> bool:
    """>= 1 accepted|edited item, no remaining ``proposed`` items and (when required) the latest critique's
    ``would_use_as_fewshot`` is true."""
    if not exportable_items(file) or any(it.status == "proposed" for it in file.items):
        return False
    if require_flag:
        c = latest_critique(campaign, file.id)
        return bool(c and c.panel.would_use_as_fewshot)
    return True


def _order_key(f: LensMarkFile) -> tuple:
    return (0 if f.system.rank is not None else 1, f.system.rank if f.system.rank is not None else 0, f.id)


def select_fewshot(campaign: Campaign, *, k: int = 6, require_flag: bool = False,
                   ids: Optional[list[str]] = None) -> list[LensMarkFile]:
    """Eligible files ordered by rank then id, stratified round-robin over ``system.verdict``, first ``k``."""
    cands = sorted((f for f in load_files(campaign, ids) if eligible(campaign, f, require_flag=require_flag)), key=_order_key)
    buckets: dict[str, list[LensMarkFile]] = {}
    for f in cands:
        buckets.setdefault(f.system.verdict or "unknown", []).append(f)
    order = sorted(buckets)          # deterministic bucket order
    chosen: list[LensMarkFile] = []
    while len(chosen) < k and any(buckets[v] for v in order):
        for v in order:
            if buckets[v] and len(chosen) < k:
                chosen.append(buckets[v].pop(0))
    return sorted(chosen, key=_order_key)


def example_markdown(file: LensMarkFile) -> str:
    """The prose card: description first (it refers to arrows by colour), then one line per arrow and a
    summary of masks / ring / notes."""
    s = file.system
    head = f"# {file.id}"
    bits = []
    if s.rank is not None:
        bits.append(f"rank {s.rank}")
    if s.verdict:
        bits.append(s.verdict)
    if s.grade:
        bits.append(f"grade {s.grade}")
    if s.theta_e.value_arcsec is not None:
        bits.append(f"theta_E ~ {s.theta_e.value_arcsec:.2f}\"" + (f" ({s.theta_e.method})" if s.theta_e.method else ""))
    if bits:
        head += " - " + ", ".join(bits)
    lines = [head, "", (s.description.strip() or "(no description)"), ""]
    items = exportable_items(file)
    arrows = [it for it in items if isinstance(it, Arrow)]
    if arrows:
        lines.append("Arrows:")
        lines += [f"- {a.color} arrow: {a.label or '(unlabelled)'}" for a in arrows]
        lines.append("")
    masks = [it for it in items if isinstance(it, MaskCircle)]
    if masks:
        by_kind: dict[str, int] = {}
        for m in masks:
            by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
        stroke = {"galaxy": "dashed", "star": "dotted", "artifact": "short-dashed"}
        parts = [f"{n} {k} ({stroke[k]})" for k, n in sorted(by_kind.items())]
        radii = sorted(m.radius_arcsec for m in masks)
        lines.append(f"Masks: {', '.join(parts)}; radii {radii[0]:.2f}-{radii[-1]:.2f}\"")
    rings = [it for it in items if isinstance(it, EinsteinRing)]
    for r in rings:
        c = ring_center(file, r)
        lines.append(f"Einstein ring: theta_E = {r.theta_e_arcsec:.2f}\" centred at [{c[0]:.3f}, {c[1]:.3f}]"
                     + (f" (on {r.center_ref})" if r.center_ref else ""))
    notes = [it for it in items if isinstance(it, TextNote)]
    for t in notes:
        lines.append(f"Note: \"{t.text}\" at [{t.pos[0]:.3f}, {t.pos[1]:.3f}]")
    if s.tags:
        lines.append("Tags: " + ", ".join(s.tags))
    return "\n".join(lines).rstrip() + "\n"


def _clear_bundle(out_dir: Path) -> None:
    for p in out_dir.iterdir():
        if p.is_file() and (_BUNDLE_FILE_RE.match(p.name) or p.name in ("manifest.json", "prompt.sha256")):
            p.unlink()


def _copy_original(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".png":
        shutil.copyfile(src, dst)
    else:
        with Image.open(src) as im:
            im.convert("RGB").save(dst, format="PNG", optimize=False)


def build_bundle(campaign: Campaign, files: list[LensMarkFile], out_dir: Path, *, k: int) -> dict[str, Any]:
    _clear_bundle(out_dir)
    examples: list[dict[str, Any]] = []
    h = hashlib.sha256()
    for n, f in enumerate(files, start=1):
        stem = f"{n:03d}-{f.id}"
        _copy_original(campaign.image_path(f.id), out_dir / f"{stem}.png")
        annot: Optional[str] = None
        if campaign.annot_path(f.id).exists() and not campaign.annot_stale(f.id, f):
            shutil.copyfile(campaign.annot_path(f.id), out_dir / f"{stem}.annot.png")
            annot = f"{stem}.annot.png"
        json_text = f.to_json()
        md_text = example_markdown(f)
        atomic_write_text(out_dir / f"{stem}.lensmark.json", json_text)
        atomic_write_text(out_dir / f"{stem}.md", md_text)
        h.update(json_text.encode("utf-8"))
        h.update(md_text.encode("utf-8"))
        examples.append({"id": f.id, "png": f"{stem}.png", "annot": annot, "json": f"{stem}.lensmark.json",
                         "md": f"{stem}.md", "rank": f.system.rank, "verdict": f.system.verdict})
    manifest = {"schema_version": SCHEMA_VERSION, "k": k, "n": len(examples), "statuses": list(EXPORT_STATUSES),
                "examples": examples, "prompt_sha256": h.hexdigest()}
    atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(out_dir / "prompt.sha256", manifest["prompt_sha256"] + "\n")
    return manifest


def load_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    with open(Path(bundle_dir) / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def export_fewshot(campaign: Campaign, *, out: str | Path | None = None, k: int = 6, require_flag: bool = False,
                   ids: Optional[list[str]] = None) -> list[Path]:
    files = select_fewshot(campaign, k=k, require_flag=require_flag, ids=ids)
    if not files:
        return []
    out_dir = export_dir(campaign, "fewshot", out)
    manifest = build_bundle(campaign, files, out_dir, k=k)
    paths = [out_dir / "manifest.json", out_dir / "prompt.sha256"]
    for ex in manifest["examples"]:
        paths += [out_dir / ex["png"], out_dir / ex["json"], out_dir / ex["md"]]
        if ex["annot"]:
            paths.append(out_dir / ex["annot"])
    return paths
