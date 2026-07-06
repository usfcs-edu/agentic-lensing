"""cgl — claude-giga-lens campaign package.

Import-order contract (inherited from foundry-i `_hmc_lib_marg.py`):

  float64 must be enabled BEFORE the first jnp array is created. Driver scripts
  that build real-data posteriors set ``os.environ['GIGALENS_X64'] = '1'`` (or
  pass ``--x64``) BEFORE importing any cgl module that touches jax. cgl modules
  honor the same env var and never flip x64 themselves after import time.

  The vendored gigalens-sean library (multinode-2025 @ 58ec9a7, UNPATCHED) is
  activated with ``cgl.paths.bootstrap_vendor()``, which must run before any
  ``import gigalens.*``. All known-defect mitigations live OUTSIDE the vendored
  tree — see ``cgl.guards`` (each guard encodes a prior real incident).

This top-level module stays import-light on purpose: no jax, no gigalens.
"""

__version__ = "0.1.0"
