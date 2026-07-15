from abc import ABC, abstractmethod
from typing import List


class Parameterized(ABC):
    """Interface for a parameterized profile of any kind.

    Attributes:
        name (str): Name of the profile
        params (:obj:`list` of :obj:`str`): List of parameter names
    """

    _name: str  # Static class level default for name
    _params: List[str]  # Static class level default for parameter names

    def __init__(self, *args, **kwargs):
        # self.constants = constants  TODO: include constants in Parametrized plus a @with_constants decorator
        self.name = self._name
        self.params = self._params.copy()  # [param for param in self._params if param not in constants]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class LightProfile(Parameterized, ABC):
    """Interface for a light profile.

    Keyword Args:
         use_lstsq (bool): Whether to use least squares to solve for linear parameters

    Attributes:
         _use_lstsq (bool): Whether to use least squares to solve for linear parameters
    """

    _amp = ""

    def __init__(self, use_lstsq=False, is_source=False, cosmo_sample=True, *args, **kwargs):
        super(LightProfile, self).__init__(*args, **kwargs)
        self._use_lstsq = use_lstsq
        # ``is_source`` / ``cosmo_sample`` are retained as INERT NO-OP kwargs: the old API
        # used them to inject a ``z_source`` / ``deflection_ratio`` param into the profile
        # (the "is_source surgery"), but geometry now lives on the scene ``Plane``, so the
        # scene path always builds profiles with ``is_source=False`` and the surgery is
        # dead. The logic was removed when the old API was dropped; the params are KEPT in
        # the signature so the many working-path profiles that pass ``is_source=is_source``
        # don't need a cascade signature change (full removal = future cleanup).
        self._is_source = is_source
        self.cosmo_sample = cosmo_sample
        self.depth = 1
        if not self.use_lstsq:
            self.params.append(self._amp)

    @property
    def use_lstsq(self):
        return self._use_lstsq

    @use_lstsq.setter
    def use_lstsq(self, use_lstsq: bool):
        """
        Arguments:
             use_lstsq (bool): Whether to use least squares to solve for linear parameters
        """
        if use_lstsq and not self.use_lstsq:  # Turn least squares on
            self.params.pop(self.params.index(self._amp))
        elif not use_lstsq and self.use_lstsq:  # Turn least squares off
            self.params.append(self._amp)
        self._use_lstsq = use_lstsq

    @property
    def is_source(self):
        return self._is_source

    @is_source.setter
    def is_source(self, is_source: bool):
        """Inert: set the flag only. The old param-list surgery (injecting
        ``z_source`` / ``deflection_ratio``) was removed with the old API; geometry now
        lives on the scene ``Plane``."""
        self._is_source = is_source

    #: Whether this profile renders to a pixel image. ``True`` for extended light; a
    #: non-rendering emitter (e.g. a point source, whose constraints are image positions /
    #: fluxes / time delays, not pixels) sets this ``False``. A ``renders=False`` component
    #: may still live in a ``Plane``'s ``light`` list — "light" means "a lensed emitter",
    #: and *observability* is decided per-dataset by ``sees`` — but the scene simulator
    #: never renders it or counts it toward the lstsq basis. Rendering is just the
    #: imaging-specific capability of the general "observed scene entity" concept.
    renders: bool = True

    def light(self, x, y, **kwargs):
        """Surface brightness at ``(x, y)``.

        Renderable profiles (``renders=True``) override this. A non-rendering profile
        leaves it raising: it is never rendered to a pixel grid, and the scene simulator
        skips ``renders=False`` light, so reaching here means something tried to render an
        emitter that has no pixel representation."""
        raise NotImplementedError(
            f"{type(self).__name__} has renders={self.renders} and does not implement "
            "light(): it is not rendered to a pixel image (its constraints enter through a "
            "non-imaging dataset). A renderable profile must override light(); a "
            "non-rendering one must never reach the render path — the scene simulator skips "
            "renders=False light, and imaging 'sees' resolution excludes it.")

    def __str__(self):
        return f"{self.name}(use_lstsq={self.use_lstsq}, is_source={self.is_source})"


class MassProfile(Parameterized, ABC):
    """Interface for a mass profile."""

    def __init__(self, *args, **kwargs):
        super(MassProfile, self).__init__(*args, **kwargs)

    @abstractmethod
    def deriv(self, x, y, **kwargs):
        """Calculates deflection angle.

        Args:
            x: :math:`x` coordinate at which to evaluate the deflection
            y: :math:`y` coordinate at which to evaluate the deflection
            **kwargs: Mass profile parameters. Each parameter must be shaped in a way that is broadcastable with x and y

        Returns:
            A tuple :math:`(\\alpha_x, \\alpha_y)` containing the deflection angle in the :math:`x` and :math:`y` directions

        """
        pass

