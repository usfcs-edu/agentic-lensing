"""
00 -- Environment sanity check (Milestone M0).

Asserts the reproduction environment is sound before any data/model work:
python >= 3.11, torch with CUDA, the 7 TITAN RTX (sm_75) cards visible, and the
AION + scientific stack importable. Writes data/results/env.json.

Run: /home2/benson/.venvs/aion/bin/python 00_env_check.py
"""

import json
import platform
import sys

import _config as C


def main():
    info = {"python": platform.python_version(), "argv": sys.argv}
    assert sys.version_info[:2] >= (3, 11), "need python >= 3.11"

    import torch

    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    assert torch.cuda.is_available(), "CUDA not available"
    info["device_count"] = torch.cuda.device_count()
    info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info["capability"] = list(torch.cuda.get_device_capability(0))
    assert tuple(info["capability"]) == (7, 5), "expected sm_75 Turing (TITAN RTX)"

    # matmul on GPU
    x = torch.randn(2048, 2048, device="cuda")
    info["matmul_finite"] = bool(torch.isfinite((x @ x).sum()).item())

    # imports
    for mod in ["aion", "datasets", "astropy", "sklearn", "xgboost", "pandas",
                "h5py", "matplotlib", "scipy"]:
        __import__(mod)
    info["imports_ok"] = True
    info["models"] = C.MODELS

    (C.RESULTS / "env.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print("ENV_OK: %d x %s (sm_%d%d), torch %s cu%s"
          % (info["device_count"], info["devices"][0], *info["capability"],
             info["torch"], info["cuda"]))


if __name__ == "__main__":
    main()
