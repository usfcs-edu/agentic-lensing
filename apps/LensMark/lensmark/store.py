"""Campaign store: a directory of cutouts with the three-files-per-image contract.

    <campaign>/lensmark.config.json      campaign defaults (+ per-image overrides)
    <campaign>/lensmark.manifest.json    DERIVED index, rebuilt by ``Campaign.write_manifest()``
    <campaign>/<id>.png|.jpg             original (never written)
    <campaign>/<id>.lensmark.json        source of truth
    <campaign>/<id>.annot.png            rendered overlay (derived; ``render.of_json_sha256`` pins it)
    <campaign>/<id>.lensmark.log.jsonl   append-only event log
    <campaign>/proposals/  critiques/  exports/

Writes are atomic (tmp + rename) and serialised per id with a lock. Every save appends a log line per
item added / removed / changed (before/after) so history is replayable without a ``revisions[]`` array.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from . import config, coords
from .model import ImageMeta, LensMarkFile, SystemBlock, ThetaE, now_iso


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def is_original_image(path: Path) -> bool:
    if path.suffix.lower() not in config.IMAGE_EXTS:
        return False
    stem = path.stem
    return not any(stem.endswith(s) for s in config.DERIVED_SUFFIXES)


class Campaign:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"campaign directory not found: {self.root}")
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.config: dict[str, Any] = self.load_config()

    # ------------------------------------------------------------------ paths
    @property
    def config_path(self) -> Path:
        return self.root / config.CAMPAIGN_CONFIG_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / config.CAMPAIGN_MANIFEST_NAME

    @property
    def proposals_dir(self) -> Path:
        return self.root / "proposals"

    @property
    def critiques_dir(self) -> Path:
        return self.root / "critiques"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    def image_path(self, image_id: str) -> Path:
        for ext in config.IMAGE_EXTS:
            p = self.root / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no original image for id {image_id!r} in {self.root}")

    def json_path(self, image_id: str) -> Path:
        return self.root / f"{image_id}.lensmark.json"

    def annot_path(self, image_id: str) -> Path:
        return self.root / f"{image_id}.annot.png"

    def log_path(self, image_id: str) -> Path:
        return self.root / f"{image_id}.lensmark.log.jsonl"

    def mask_path(self, image_id: str) -> Path:
        return self.root / f"{image_id}.mask.png"

    # ------------------------------------------------------------------ config
    def load_config(self) -> dict[str, Any]:
        cfg = json.loads(json.dumps(config.CAMPAIGN_DEFAULTS))
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                user = json.load(f)
            for k, v in user.items():
                cfg[k] = v
        cfg.setdefault("overrides", {})
        return cfg

    def save_config(self, cfg: Optional[dict[str, Any]] = None) -> None:
        if cfg is not None:
            self.config = cfg
        atomic_write_text(self.config_path, json.dumps(self.config, indent=2, ensure_ascii=False) + "\n")

    def override(self, image_id: str) -> dict[str, Any]:
        return dict(self.config.get("overrides", {}).get(image_id, {}))

    # ------------------------------------------------------------------ listing
    def list_ids(self) -> list[str]:
        ids = sorted({p.stem for p in self.root.iterdir() if p.is_file() and is_original_image(p)})
        return ids

    def image_size(self, image_id: str) -> tuple[int, int]:
        with Image.open(self.image_path(image_id)) as im:
            return im.size

    def lock(self, image_id: str) -> threading.Lock:
        with self._locks_guard:
            if image_id not in self._locks:
                self._locks[image_id] = threading.Lock()
            return self._locks[image_id]

    # ------------------------------------------------------------------ files
    def cutout_arcsec_for(self, image_id: str) -> tuple[float, str]:
        ov = self.override(image_id)
        if "cutout_arcsec" in ov:
            return float(ov["cutout_arcsec"]), "override"
        return float(self.config["cutout_arcsec"]), str(self.config.get("cutout_arcsec_source", "config"))

    def new_file(self, image_id: str) -> LensMarkFile:
        """A fresh, unsaved LensMarkFile initialised from the image + campaign config."""
        path = self.image_path(image_id)
        W, H = self.image_size(image_id)
        cutout, source = self.cutout_arcsec_for(image_id)
        ov = self.override(image_id)
        image = ImageMeta(
            file=path.name, sha256=sha256_file(path), width=W, height=H,
            cutout_arcsec=cutout, pixel_scale_arcsec=coords.pixel_scale(W, cutout),
            native_pixel_scale_arcsec=ov.get("native_pixel_scale_arcsec", self.config.get("native_pixel_scale_arcsec")),
            array_origin=self.config.get("array_origin", "upper"),
            north_up=bool(self.config.get("north_up", True)), east_left=bool(self.config.get("east_left", True)),
            survey=ov.get("survey", self.config.get("survey")),
            scale_source=source if source in ("config", "override", "header", "assumed") else "config",
        )
        if ov.get("ra_deg") is not None and ov.get("dec_deg") is not None:
            image.wcs = {"ra_deg": float(ov["ra_deg"]), "dec_deg": float(ov["dec_deg"]), "rot_deg": 0.0}  # type: ignore[assignment]
        system = SystemBlock(object_id=ov.get("object_id"), rank=ov.get("rank"),
                             theta_e=ThetaE(value_arcsec=ov.get("theta_e_ref_arcsec"),
                                            method="reference" if ov.get("theta_e_ref_arcsec") else None))
        f = LensMarkFile(id=image_id, image=image, system=system)
        f.provenance.log = self.log_path(image_id).name
        return f

    def exists(self, image_id: str) -> bool:
        return self.json_path(image_id).exists()

    def load(self, image_id: str) -> Optional[LensMarkFile]:
        p = self.json_path(image_id)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as fh:
            return LensMarkFile.from_json(fh.read())

    def load_or_new(self, image_id: str) -> LensMarkFile:
        return self.load(image_id) or self.new_file(image_id)

    def save(self, image_id: str, file: LensMarkFile, *, actor: str = "ui", source: str = "ui",
             before: Optional[LensMarkFile] = None, touch_modified: bool = True) -> LensMarkFile:
        """Validate, write atomically, append diff events. Does NOT render (the server does that)."""
        if file.id != image_id:
            raise ValueError(f"file id {file.id!r} != {image_id!r}")
        with self.lock(image_id):
            if before is None:
                before = self.load(image_id)
            if touch_modified:
                file.modified = now_iso()
            if file.provenance.log is None:
                file.provenance.log = self.log_path(image_id).name
            atomic_write_text(self.json_path(image_id), file.to_json())
            for ev in diff_events(before, file):
                self.append_log(image_id, actor=actor, source=source, **ev)
        return file

    def append_log(self, image_id: str, **event: Any) -> None:
        event = {"ts": now_iso(), **event}
        with open(self.log_path(image_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def read_log(self, image_id: str) -> list[dict[str, Any]]:
        p = self.log_path(image_id)
        if not p.exists():
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    # ------------------------------------------------------------------ derived
    def annot_stale(self, image_id: str, file: Optional[LensMarkFile] = None) -> bool:
        file = file or self.load(image_id)
        if file is None or not self.annot_path(image_id).exists():
            return True
        return file.render is None or file.render.of_json_sha256 != file.content_sha256()

    def summary(self, image_id: str) -> dict[str, Any]:
        path = self.image_path(image_id)
        file = self.load(image_id)
        W, H = self.image_size(image_id)
        cutout, source = self.cutout_arcsec_for(image_id)
        n_items = len(file.items) if file else 0
        by_status: dict[str, int] = {}
        if file:
            for it in file.items:
                by_status[it.status] = by_status.get(it.status, 0) + 1
        return {
            "id": image_id, "file": path.name, "width": W, "height": H,
            "cutout_arcsec": file.image.cutout_arcsec if file else cutout,
            "scale_source": file.image.scale_source if file else source,
            "has_json": file is not None, "has_annot": self.annot_path(image_id).exists(),
            "annot_stale": self.annot_stale(image_id, file),
            "n_items": n_items, "by_status": by_status,
            "grade": file.system.grade if file else None,
            "verdict": file.system.verdict if file else None,
            "theta_e_arcsec": file.system.theta_e.value_arcsec if file else self.override(image_id).get("theta_e_ref_arcsec"),
            "rank": file.system.rank if file else self.override(image_id).get("rank"),
            "modified": file.modified if file else None,
            "n_proposals": len(file.provenance.proposal_runs) if file else 0,
        }

    def manifest(self) -> list[dict[str, Any]]:
        return [self.summary(i) for i in self.list_ids()]

    def write_manifest(self) -> list[dict[str, Any]]:
        rows = self.manifest()
        atomic_write_text(self.manifest_path, json.dumps(
            {"schema_version": "lensmark-manifest/1.0", "generated": now_iso(), "root": str(self.root),
             "images": rows}, indent=2, ensure_ascii=False) + "\n")
        return rows


# ---------------------------------------------------------------------------- diff -> log events
def _item_map(f: Optional[LensMarkFile]) -> dict[str, dict[str, Any]]:
    if f is None:
        return {}
    return {it.id: it.model_dump(mode="json", exclude_none=True) for it in f.items}


def diff_events(before: Optional[LensMarkFile], after: LensMarkFile) -> list[dict[str, Any]]:
    """One event per added/removed/changed item plus one for system-block changes."""
    b, a = _item_map(before), _item_map(after)
    events: list[dict[str, Any]] = []
    for iid in a:
        if iid not in b:
            events.append({"op": "add", "item_id": iid, "after": a[iid]})
        elif a[iid] != b[iid]:
            events.append({"op": "update", "item_id": iid, "before": b[iid], "after": a[iid]})
    for iid in b:
        if iid not in a:
            events.append({"op": "delete", "item_id": iid, "before": b[iid]})
    bs = before.system.model_dump(mode="json", exclude_none=True) if before else None
    as_ = after.system.model_dump(mode="json", exclude_none=True)
    if bs != as_:
        events.append({"op": "update", "item_id": "$system", "before": bs, "after": as_})
    if before is None:
        events.insert(0, {"op": "create", "item_id": "$file"})
    return events
