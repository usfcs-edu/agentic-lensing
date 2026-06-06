#!/usr/bin/env python3
"""00_env_check.py — validate both venvs, the 6-GPU allowlist, the thermal-card
exclusion, and the reuse surface, before any expensive step.

Run with the claudenet venv:
    /home2/benson/.venvs/claudenet/bin/python 00_env_check.py
Writes data/env.json. Hard-asserts the things later phases depend on.
"""
from __future__ import annotations

import json
import subprocess
import sys

import _clib as C


def gpu_table():
    """nvidia-smi GPUs in PCI-bus-id order: [{index, bus_id, name, temp}]."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id,name,temperature.gpu",
         "--format=csv,noheader"],
        env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PATH": "/usr/bin:/bin"},
        text=True)
    rows = []
    for ln in out.strip().splitlines():
        idx, bus, name, temp = [s.strip() for s in ln.split(",")]
        rows.append({"index": int(idx), "bus_id": bus, "name": name, "temp_c": int(temp)})
    return rows


def main():
    info = {"ok": True, "errors": []}

    # --- claudenet venv: torch / cuda / sm_75 -------------------------------
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device_count"] = int(torch.cuda.device_count())
    assert torch.cuda.is_available(), "CUDA not available in claudenet venv"
    cap = torch.cuda.get_device_capability(0)
    info["sm"] = f"{cap[0]}.{cap[1]}"
    assert cap == (7, 5), f"expected Turing sm_75, got {cap}"
    assert torch.cuda.device_count() >= 6, "need >=6 visible GPUs"

    for mod in ("timm", "sklearn", "astropy", "pandas", "pyarrow", "lenstronomy", "scipy"):
        __import__(mod)
    info["imports_ok"] = True

    # --- GPU table + thermal-card (GPU index 1) exclusion -------------------
    gpus = gpu_table()
    info["gpus"] = gpus
    info["gpus_usable"] = C.GPUS
    bus_by_idx = {g["index"]: g["bus_id"] for g in gpus}
    info["thermal_card"] = {"index": 1, "bus_id": bus_by_idx.get(1)}
    assert 1 not in C.GPUS, "GPU 1 (thermal throttler) must be excluded"
    assert all(i in bus_by_idx for i in C.GPUS), "a usable GPU index is missing from nvidia-smi"

    # --- aion venv launches `import aion` (needed by 11_embed_aion) ---------
    r = subprocess.run([C.AION_PY, "-c", "import aion; print('aion-ok')"],
                       capture_output=True, text=True)
    info["aion_venv_ok"] = (r.returncode == 0 and "aion-ok" in r.stdout)
    if not info["aion_venv_ok"]:
        info["errors"].append(f"aion venv import failed: {r.stderr.strip()[:300]}")

    # --- reuse surface: model param counts ---------------------------------
    M = C.models()
    sh = M["ShieldedDeepLens"](in_channels=3, **C.CFG194)
    n_sh = sum(p.numel() for p in sh.parameters())
    info["shielded194k_params"] = n_sh
    assert 190_000 <= n_sh <= 200_000, f"shielded param count {n_sh} != ~194,433"
    ef = M["EfficientNetV2Lens"](pretrained=False)
    n_ef = sum(p.numel() for p in ef.parameters())
    info["effnet_params"] = n_ef
    assert 20_400_000 <= n_ef <= 20_650_000, f"effnet param count {n_ef} != ~20.54M"

    import _trainlib as TL  # noqa: F401  (symlink import works)
    import _scorelib as SL  # noqa: F401
    info["reuse_import_ok"] = True

    if info["errors"]:
        info["ok"] = False
    (C.DATA / "env.json").write_text(json.dumps(info, indent=2))

    print(f"[env] torch {info['torch']} cuda={info['cuda_available']} "
          f"n_gpu={info['device_count']} sm={info['sm']}")
    print(f"[env] usable GPUs {C.GPUS}; thermal card idx1 bus={info['thermal_card']['bus_id']}")
    print(f"[env] aion venv import: {'OK' if info['aion_venv_ok'] else 'FAILED'}")
    print(f"[env] shielded={n_sh:,} effnet={n_ef:,} reuse_import=OK")
    print(f"[env] {'OK' if info['ok'] else 'HAD ERRORS: ' + '; '.join(info['errors'])}")
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
