"""Legacy ``PhysicalModel`` — removed with the old gigalens API.

The scene path uses :class:`gigalens.jax.scene.LensModel` instead. Nothing on the working
path imports this module. The class is kept as an import-safe stub that raises on
CONSTRUCTION (so a module that merely imports it — e.g. the anchor golden regenerator —
still imports). Restore from git b82397c if the old class is genuinely needed.
"""


class PhysicalModel:
    """Removed with the old gigalens API — use :class:`gigalens.jax.scene.LensModel`."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "gigalens.jax.physical_model.PhysicalModel was removed with the old gigalens "
            "API; use gigalens.jax.scene.LensModel. Restore from git b82397c if needed."
        )
