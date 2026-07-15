"""cgl2 — claude-giga-lens-linus campaign package.

x64 bootstrap contract: importing cgl2 sets JAX_ENABLE_X64 from the environment
BEFORE jax.numpy is touched (their project-standards §8 pattern + our incident).
Operator scripts import cgl2.paths first and call the guards they need.
"""
import os

if os.environ.get("GIGALENS_X64", os.environ.get("JAX_ENABLE_X64", "1")) == "1":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
    except ImportError:  # pragma: no cover
        pass
