from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np

import gigalens.model


_ALLOWED_PRECISIONS = (np.dtype("float32"), np.dtype("float64"))


def _normalize_precision(value, field):
    """Normalize a precision spec to a numpy float dtype (or ``None``).

    Accepts a string (``"float32"``/``"float64"``), a numpy/jax dtype or float scalar
    type (``np.float64``, ``jnp.float32`` — these are the same numpy types), or ``None``
    for "no override". Storing the dtype lets consumers cast with a bare ``astype`` and
    drop the string-dispatch ``if`` ladder; ``None`` is preserved as a real third state
    (inherit the ambient dtype). Anything other than float32/float64 raises (no silent
    precision default)."""
    if value is None:                      # np.dtype(None) is float64 -- guard first.
        return None
    try:
        dt = np.dtype(value)
    except TypeError as e:
        raise ValueError(
            f"{field} must be 'float32'/'float64', a float32/float64 dtype, or None; "
            f"got {value!r}") from e
    if dt not in _ALLOWED_PRECISIONS:
        raise ValueError(
            f"{field} must be 'float32'/'float64' (or the matching dtype) or None; "
            f"got {value!r}")
    return dt


@dataclass
class SimulatorConfig:
    """Holds parameters for simulation.

    Attributes:
        delta_pix (float): The pixel scale (i.e., the angular resolution between adjacent pixels)
        num_pix (int or tuple of int): The number of pixels on the grid in the x and y directions. If is an int
            then the grid shape is (num_pix, num_pix). If a tuple, then the grid shape is (num_pix[0], num_pix[1]).
            This should match the shape of the observed image (image.shape).
        supersample (int): Supersampling factor
        kernel (:obj:`numpy.array`, optional): The point spread function with which to convolve simulated images
        transform_pix2angle (:obj:`numpy.array`, optional): An array mapping indices on the coordinate grid to
            angular units (RA and DEC)
        pix_region (:obj:`numpy.array`, optional): A mask representing the region of the image that will be
            simulated. True values are included. If None, the entire image will be simulated.
        angular_shift (tuple of float, optional): The angular shift of the center of the grid in arcseconds
    """

    delta_pix: float
    num_pix: Union[int, Tuple[int, int]]
    supersample: Optional[int] = 1
    kernel: Optional[Any] = None
    transform_pix2angle: Optional[np.array] = None
    pix_region: Optional[np.array] = None
    angular_shift: Optional[Tuple[float, float]] = (0., 0.)
    # Likelihood precision (None -> "float64"; "float64" requires jax_enable_x64):
    #   "float32" : baseline, byte-identical to the float32 pipeline.
    #   "float64" : full float64 forward + reduction (default). Removes the basis noise
    #               floor; ~2x forward memory/compute. Required for high-n_max shapelet
    #               sampling (see diagnosis 2026-06).
    #   Stored as a ``numpy.dtype`` (or None); a string passed to the constructor is
    #   normalized in ``__post_init__`` (see ``_normalize_precision``).
    likelihood_precision: Optional[Any] = None
    # PSF convolution precision override (independent of likelihood_precision):
    #   None      : convolve in the basis dtype (default; byte-identical to before).
    #   "float32" : downcast the (f64) basis + kernel to float32 for the convolution
    #               arithmetic only, then cast the result back to the basis dtype before
    #               pooling / the linear solve / the reduction. The PSF convolution is a
    #               well-conditioned linear smoothing, so float32 here is numerically
    #               benign (~1e-7 relative), but on Ampere it moves the FLOP-dominant
    #               per-component conv off the slow FP64 path (9.7 -> 19.5+ TFLOPS).
    #               Basis generation, gram/solve and reduction stay in float64.
    conv_precision: Optional[Any] = None
    # Light-basis precision override (independent of likelihood_precision):
    #   None      : render the light basis in the param/grid-promoted dtype (default;
    #               float64 under a float64 likelihood — byte-identical to before).
    #   "float32" : cast the deflected positions + light params to float32 for the
    #               per-component light-basis evaluation (and its VJP) only. The basis
    #               and PSF convolution then run in float32; the gram/solve and the
    #               chi^2 reduction are promoted back to float64 automatically by the
    #               float64 noise/observed arrays. Profiling (wip/profile_scene_likelihood)
    #               shows the basis generation + its backward pass dominate the gradient
    #               and the kernel is ~co-limited by HBM bandwidth and half-rate Ampere
    #               FP64; float32 here roughly halves the dominant intermediate's bytes.
    #               WHETHER this is scientifically acceptable (MCLMC step-size adaptation
    #               unchanged; gram conditioning at high n_max) is an OPEN question being
    #               tested — do not enable by default.
    basis_precision: Optional[Any] = None
    # Remat (jax.checkpoint) of the basis+conv+pool pipeline in lstsq_simulate:
    # the backward pass RECOMPUTES the pipeline instead of storing its
    # supersampled/complex intermediates. Mathematically exact (identical ops;
    # gates: wip/validate_map_throughput.py + wip/validate_remat_default.py).
    #   False (default): fastest gradient evals. A default flip to True was
    #        attempted 2026-07-09 and REJECTED by its pre-registered gates:
    #        at compute saturation the recompute no longer hides in idle memory
    #        bandwidth — vmapped 8-chain sampler shape pays +8.3% (ss2) and the
    #        fused ss>=4 single eval +23% (compute-profiling C-17).
    #   True : -40-46% peak memory and 2x feasible batch — worth it ONLY for
    #        MEMORY-bound batched MAP/SVI (C-16); enable per stage, e.g.
    #        dataclasses.replace(cfg, remat_basis=True) at the MAP call site.
    remat_basis: bool = False

    def __post_init__(self):
        # Normalize precision specs (str / dtype / None) to numpy dtypes once, at
        # construction, so the simulator can cast with a bare ``astype`` (no string
        # dispatch). Strings are still accepted as constructor input.
        self.likelihood_precision = _normalize_precision(
            self.likelihood_precision, "likelihood_precision")
        self.conv_precision = _normalize_precision(self.conv_precision, "conv_precision")
        self.basis_precision = _normalize_precision(self.basis_precision, "basis_precision")


class LensWCS:
    def __init__(self, n, supersample=1, transform_pix2angle=None, pix_scale=1., shift=(0., 0.)):

        if transform_pix2angle is None:
            transform_pix2angle = np.eye(2) * pix_scale
        self.transform_pix2angle = transform_pix2angle / supersample
        self.transform_angle2pix = np.linalg.inv(transform_pix2angle)

        if isinstance(n, int):
            self.n_y, self.n_x = n, n
        else:
            self.n_y, self.n_x = n

        self.supersample = supersample

        low_x = -(self.n_x * self.supersample - 1) / 2
        low_y = -(self.n_y * self.supersample - 1) / 2

        self.radec_at_xy_0 = np.squeeze((self.transform_pix2angle @ ([[low_x], [low_y]]))) + np.asarray(shift)

    def pix2angle(self, x, y):
        radec = np.einsum('ij,i...->...j',
                          self.transform_pix2angle,
                          np.concatenate([[x], [y]])) + self.radec_at_xy_0
        radec = np.swapaxes(radec, -1, 0).astype(np.float64) # float64 now
        return radec[0].T, radec[1].T

    def angle2pix(self, ra, dec):
        radec = np.swapaxes(np.concatenate([[ra], [dec]]), -1, 0) - self.radec_at_xy_0
        pixel =  np.einsum('ij,...i->j...', self.transform_angle2pix, radec).astype(np.float64) # float64 now
        return pixel * self.supersample

    def pixel_grid(self):
        x, y = np.arange(self.n_x * self.supersample), np.arange(self.n_y * self.supersample)
        X, Y = np.meshgrid(x, y)
        return self.pix2angle(X, Y)


class LensSimulatorInterface(ABC):
    """
    A class to simulate batches of lenses given a physical model and camera configuration options (i.e., pixel scale,
    number of pixels, point spread function, etc.).

    Attributes:
        phys_model (:obj:`~gigalens.model.PhysicalModel`): The physical model that generated the lensing system. All
            parameters ``z`` that are passed to the simulation methods are expected to correspond to this physical
            model.
        sim_config (:obj:`~gigalens.simulator.SimulatorConfig`): Camera configuration settings
        bs (int): The number of lenses to simulate in parallel
    """

    def __init__(
        self,
        phys_model: gigalens.model.PhysicalModelBase,
        sim_config: SimulatorConfig,
        bs: int,
    ):
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.bs = bs
        self.wcs = LensWCS(n=sim_config.num_pix, supersample=sim_config.supersample,
                           transform_pix2angle=sim_config.transform_pix2angle, pix_scale=sim_config.delta_pix,
                           shift=sim_config.angular_shift)

    @abstractmethod
    def simulate(
        self,
        params: Dict[str, Dict[str, Dict]],
    ):
        """Simulates lenses with physical parameters ``params``.

        Args:
            params (:obj:`tuple` of :obj:`list` of :obj:`dict`): Parameters of the lensing system to simulate, with
                format (``lens_params``, ``lens_light_params``, ``source_light_params``)

        Returns:
            Simulated lenses
        """
        pass

    @abstractmethod
    def lstsq_simulate(
        self,
        params: Dict[str, Dict[str, Dict]],
        observed_image,
        err_map,
    ):
        """Simulates lenses and fits for their linear parameters based on the observed data.

        Args:
            params (:obj:`tuple` of :obj:`list` of :obj:`dict`): Parameters of the lensing system to simulate, with
                format (``lens_params``, ``lens_light_params``, ``source_light_params``)
            observed_image: The observed image, needed for solving the linear parameters
            err_map: Noise variance map, needed for solving the linear parameters

        Returns:
            Simulated images that have the linear parameters set to the optimal values, minimizing the chi-squared
            of the residual using ``err_map`` as the variance.
        """
        pass
