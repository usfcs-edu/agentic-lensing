---
name: reference-gigalens-env
description: "Working JAX/CUDA env for GIGA-Lens reproductions on /raid/benson host — venv path, key versions, available GPUs"
metadata:
  type: reference
---

The working environment for GIGA-Lens reproductions on the `/raid/benson` host is the venv at `/home/benson/.venvs/gigalens/` (Python 3.13). Activate with `source ~/.venvs/gigalens/bin/activate` or use `/home/benson/.venvs/gigalens/bin/python` directly.

Key pinned versions as of 2026-05-25 Phase-0 setup:
- `jax==0.6.2` with `jax[cuda12]` (10 CUDA devices visible: 8× NVIDIA A16 15 GB + 2× NVIDIA L4 23 GB)
- `tensorflow==2.20.0` + `tensorflow-probability==0.25.0` (Greg got TF 2.20 working on aarch64 despite no PyPI wheels — relevant for any TF-backend gigalens work)
- `lenstronomy==1.14.0`, `optax==0.2.8`, `numpy==2.4.6`, `objax==1.8.0`
- `gigalens 1.2.1` editable from `/raid/benson/lensing-repos/gigalens/` (multi-node branch, GIGA-Lens 2.0)
- JupyterLab 4.5 installed

Active repo: `/raid/benson/lensing-repos/gigalens/` on branch `multi-node`; frozen Gu-2022 baseline at `/raid/benson/lensing-repos/gigalens-archive/`. Other branches worth comparing: `xh-dev` (Huang's personal), `multinode-2025`, `single-node`.

Smoke-test recipe (cd to the gigalens repo first so `./src/gigalens/assets/psf.npy` resolves):
```python
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
import numpy as np
kernel = np.load('./src/gigalens/assets/psf.npy').astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel)
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()],
                           [sersic.SersicEllipse(use_lstsq=False)],
                           [sersic.SersicEllipse(use_lstsq=False)])
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
```
Benchmark on a single A16 with JIT cached: ~1.2 ms per forward sim.

The `bootstrapping` pitfall: this venv was created without pip; if `bin/pip` is missing, run `~/.venvs/gigalens/bin/python -m ensurepip --upgrade`.

Related: [[project-huang-lensing]], [[reference-paper-corpus]].
