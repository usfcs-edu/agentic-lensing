import gigalens.profile
from gigalens.physicality import Domain


class PointSource(gigalens.profile.LightProfile):
    """A point source on a (source) plane: intrinsic flux ``amp`` (position is derived).

    The only sampled parameter is ``amp``, the intrinsic (unlensed) point-source flux, so
    ``params["planes"][src]["light"][k]`` carries just ``amp``. The **source position is
    NOT a parameter**: faithfully to the reference pipeline (Baltazar et al. 2026,
    ``JAX-H0-SN_demo.ipynb`` cell 28), it is *derived* inside the likelihood as the mean
    of the delensed image positions, and used only for the time-delay Fermat geometry.
    Sampling it (with a prior) would change the parameterization and the compactness
    regularization, so it is deliberately left out. All constraints (image positions /
    magnification-fluxes / time delays) enter the likelihood through
    :class:`gigalens.jax.point_source.PointSourceLikelihoodTerm`, never through pixels.

    It subclasses :class:`~gigalens.profile.LightProfile` because "light" in the scene
    means "a lensed emitter", not "a thing that renders to pixels" — and a point source is
    a compact (delta) emitter that gets lensed exactly like an extended source. It sets
    ``renders = False``: it lives in a ``Plane``'s ``light`` list and gets its ``amp`` param
    slot from the base, but the scene simulator never renders it or counts it toward the
    lstsq basis, and imaging ``sees`` resolution excludes it. It therefore does NOT
    implement :meth:`~gigalens.profile.LightProfile.light` (the base leaves it raising).

    ``amp`` is sampled in forward mode (``use_lstsq=False``); it is the ``L_F`` denominator.
    ``use_lstsq=True`` (solving it as a linear amplitude) is meaningless here — a point
    source has no extended basis to solve against — so forward mode is the only sensible use.
    """

    _name = "POINT_SOURCE"
    _params = []           # no shape/position params: the source position is DERIVED
    _amp = "amp"
    renders = False  # a point source is a lensed emitter, but not rendered to pixels
    _domains = {
        # amp enters the flux term as amp**2, so only |amp| matters; it is NOT bounded
        # positive here to avoid flagging a symmetric prior (e.g. Normal(1, 0.1)). amp==0
        # is degenerate (L_F divides by amp**2) but that is a construction-time concern,
        # not a per-sample physicality box.
        "amp": Domain(rationale=(
            "intrinsic point-source flux; enters L_F as amp**2 so only |amp| matters. "
            "amp==0 is degenerate (L_F divides by amp**2).")),
    }

    def __init__(self, use_lstsq=False, is_source=True, **kwargs):
        super(PointSource, self).__init__(use_lstsq=use_lstsq, is_source=is_source, **kwargs)
        # No extended basis: a point source stacks no lstsq rows and adds no pixels.
        self.depth = 0

    # No light() override: renders=False, so the scene simulator never renders this and the
    # base LightProfile.light (which raises) is never reached on the supported path. A real
    # lensed point source (quasar/SN) would deposit PSF-convolved flux at its image
    # positions; PSF injection into an ImageData is genuine future joint-mode work.
