"""Self-consistent drizzle mock-trio machinery (P1a, script 05).

Geometry (r=3 INTEGER design; the real fine skycell is r=3.21 but integer r
makes every operator exact):
  * fine scene grid: 214^2 @ 0.04" (render grid; PSF-convolved by gigalens),
  * NINE native exposures: 70^2 @ 0.12", frame f = 3x3 block-sum of the fine
    scene starting at integer fine offset OFFSETS_FINE[f] (detector
    pixelation), iid noise per native pixel (gu-2022 add_noise recipe),
  * drizzle back (square kernel, pixfrac=1, r=3: each native drop covers
    exactly 3x3 fine pixels) onto the fine grid; the region covered by ALL
    frames is EXACTLY the 208^2 window [2:210)^2 -> the fine product,
  * binned product: 2x2 block-sum -> 104^2 @ 0.08".

OFFSET DEVIATION (documented; the plan specified 3 exposures at
{(0,0),(1,2),(2,1)}): a 3-frame integer-dither stack is PERIOD-3
SHIFT-VARIANT — provably not a convolution (the shared-block pair counts at
diagonal lags depend on position mod 3) — so no effective fine PSF exists
and the pre-registered render-check gate fails at 2-27 sigma (measured;
data/mocks_report_3frame.json), i.e. E1 fits would carry an irreducible
simulator-mismatch floor (the gu-2022 cross-simulator lesson). Dithering
over ALL NINE sub-pixel phases {0,1,2}^2 makes the stacked
pixelation+drizzle operator EXACTLY the separable 3x3-tent convolution
(so PSF_eff is exact) while keeping the same noise-correlation scale (the
1-D stacked kernel is the identical tent (3-|d|)/3).

ONE-IMPLEMENTATION RULE: the drizzle-back operator is built from
cgl.noise.drizzle_overlap_matrix_1d — the same enumeration that produces the
noise-kernel drizzle anchors — as the separable pair (D_y, D_x) of
row-normalized 1-D overlap matrices per frame (tests/test_drizzle.py checks
the r=3 integer case equals the exact 3x3-average construction, and the
stacked noise kernel below equals cgl.noise.drizzle_acf at the same offsets).

Covariances are EXACT: per-pixel variance propagated through the operator
matrices; the stationary correlation kernel from cgl.noise.drizzle_acf with
the actual offsets (exact for constant sigma; support |lag|_inf <= 2 fine,
<= 1 binned — block sharing is local for integer r).

``sample_truth`` is PORTED WITH ATTRIBUTION from
/raid/benson/git/agentic-lensing/reproductions/gu-2022/01_gen_mocks.py
(the Gu et al. 2022 Eq. (8) SIMULATION distribution) — verbatim except rng
seeding is the caller's.
"""
from __future__ import annotations

import numpy as np

from cgl.noise import (
    binned_kernel_from_fine,
    block_sum,
    drizzle_acf,
    drizzle_overlap_matrix_1d,
)

FINE_PIX = 0.04
R_INT = 3
NATIVE_PIX = FINE_PIX * R_INT           # 0.12"
FINE_RENDER = 214                        # render grid (covers all offsets)
N_NATIVE = 70                            # native frame side
# all nine sub-pixel phases (see OFFSET DEVIATION above); the plan's three
# Latin-square offsets first, then the remaining six
OFFSETS_FINE = ((0, 0), (1, 2), (2, 1),
                (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 2))
CROP0, CROP1 = 2, 210                    # product window: exactly 208^2
N_FINE = CROP1 - CROP0                   # 208
SIGMA_BKG = 0.2                          # gu-2022 noise constants (per native px)
EXP_TIME = 100.0


# --------------------------------------------------------------------------- #
# truth sampler — PORTED WITH ATTRIBUTION from gu-2022/01_gen_mocks.py
# --------------------------------------------------------------------------- #
def sample_truth(rng):
    """Return (gigalens_truth_list, flat_dict) from the Gu+22 Eq.(8) sim dist.

    Verbatim port of gu-2022/01_gen_mocks.py::sample_truth (attribution in
    the module docstring)."""
    def trunc_normal(mu, sd, lo, hi):
        while True:
            x = rng.normal(mu, sd)
            if lo <= x <= hi:
                return float(x)

    theta_E = float(np.exp(rng.normal(np.log(1.25), 0.25)))
    gamma = trunc_normal(2.0, 0.25, 1.0, 3.0)
    e1 = float(rng.normal(0.0, 0.1))
    e2 = float(rng.normal(0.0, 0.1))
    cx = float(rng.normal(0.0, 0.05))
    cy = float(rng.normal(0.0, 0.05))
    g1 = float(rng.normal(0.0, 0.03))
    g2 = float(rng.normal(0.0, 0.03))
    Rl = float(np.exp(rng.normal(np.log(1.6), 0.15)))
    nl = float(rng.uniform(2.0, 6.0))
    le1 = trunc_normal(0.0, 0.05, -0.15, 0.15)
    le2 = trunc_normal(0.0, 0.05, -0.15, 0.15)
    lcx = float(rng.normal(0.0, 0.01))
    lcy = float(rng.normal(0.0, 0.01))
    Il = float(np.exp(rng.normal(np.log(300.0), 0.3)))
    Rs = float(np.exp(rng.normal(np.log(0.25), 0.15)))
    ns = float(rng.uniform(0.5, 4.0))
    se1 = trunc_normal(0.0, 0.15, -0.5, 0.5)
    se2 = trunc_normal(0.0, 0.15, -0.5, 0.5)
    scx = float(rng.normal(0.0, 0.25))
    scy = float(rng.normal(0.0, 0.25))
    Is = float(np.exp(rng.normal(np.log(150.0), 0.5)))

    truth = [
        [
            {"theta_E": theta_E, "gamma": gamma, "e1": e1, "e2": e2,
             "center_x": cx, "center_y": cy},
            {"gamma1": g1, "gamma2": g2},
        ],
        [
            {"R_sersic": Rl, "n_sersic": nl, "e1": le1, "e2": le2,
             "center_x": lcx, "center_y": lcy, "Ie": Il},
        ],
        [
            {"R_sersic": Rs, "n_sersic": ns, "e1": se1, "e2": se2,
             "center_x": scx, "center_y": scy, "Ie": Is},
        ],
    ]
    flat = dict(
        theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=cx, center_y=cy,
        gamma1=g1, gamma2=g2,
        ll_R_sersic=Rl, ll_n_sersic=nl, ll_e1=le1, ll_e2=le2,
        ll_center_x=lcx, ll_center_y=lcy, ll_Ie=Il,
        src_R_sersic=Rs, src_n_sersic=ns, src_e1=se1, src_e2=se2,
        src_center_x=scx, src_center_y=scy, src_Ie=Is,
    )
    return truth, flat


# --------------------------------------------------------------------------- #
# the pixelation + drizzle pipeline
# --------------------------------------------------------------------------- #
class DrizzleMockPipeline:
    """Fine scene (214^2) -> 3 native exposures (70^2) -> drizzled fine 208^2.

    All operators exact; the drizzle-back matrices come from the SAME 1-D
    overlap enumeration as the noise-kernel anchors (one implementation).
    """

    def __init__(self):
        self.offsets = OFFSETS_FINE
        # per-frame separable drizzle-back matrices (N_FINE, N_NATIVE):
        # product fine pixel j (global g = j + CROP0) vs native pixel k of
        # the frame whose native grid starts at global fine index off.
        self._D = []
        for (oy, ox) in self.offsets:
            self._D.append((self._axis_matrix(oy), self._axis_matrix(ox)))

    @staticmethod
    def _axis_matrix(off):
        """Row-normalized 1-D overlap matrix for one axis / one frame."""
        A, ks = drizzle_overlap_matrix_1d(
            R_INT, 1.0, n_out=FINE_RENDER, phase=off / float(R_INT))
        rows = A.sum(axis=1)
        At = A / rows[:, None]
        # rows: global fine g in [CROP0, CROP1); columns: native k in [0, N)
        col0 = int(np.searchsorted(ks, 0))
        D = At[CROP0:CROP1, col0:col0 + N_NATIVE]
        # every product row must be fully covered by this frame's natives
        assert np.allclose(D.sum(axis=1), 1.0, atol=1e-12)
        return D

    # ---- forward ------------------------------------------------------------
    def natives_from_fine(self, scene_fine):
        """Detector pixelation: 3x3 block-sum of the fine scene per frame."""
        assert scene_fine.shape == (FINE_RENDER, FINE_RENDER)
        nats = []
        for (oy, ox) in self.offsets:
            sub = scene_fine[oy:oy + 3 * N_NATIVE, ox:ox + 3 * N_NATIVE]
            nats.append(block_sum(sub, R_INT))
        return nats

    def drizzle(self, natives):
        """Equal-weight drizzle of the native frames onto the 208^2 window.

        Output units: flux per FINE pixel (native flux / r^2 within a drop).
        """
        out = np.zeros((N_FINE, N_FINE))
        for (Dy, Dx), nat in zip(self._D, natives):
            out += Dy @ nat @ Dx.T
        return out / (len(natives) * R_INT ** 2)

    def __call__(self, scene_fine):
        return self.drizzle(self.natives_from_fine(scene_fine))

    # ---- exact noise propagation ---------------------------------------------
    def fine_var(self, err_nats):
        """Exact per-pixel variance of the drizzled stack for iid native
        noise with per-pixel sigma maps err_nats (one per frame)."""
        var = np.zeros((N_FINE, N_FINE))
        for (Dy, Dx), err in zip(self._D, err_nats):
            var += (Dy ** 2) @ (err ** 2) @ (Dx ** 2).T
        return var / (len(err_nats) * R_INT ** 2) ** 2

    def fine_rho(self, max_lag=4):
        """Exact stationary correlation of the drizzled stack (constant-sigma
        limit) — cgl.noise.drizzle_acf at the actual offsets."""
        offs = np.asarray(self.offsets, dtype=np.float64) / R_INT
        return drizzle_acf(R_INT, 1.0, len(self.offsets), offs,
                           max_lag=max_lag)["rho"]

    def binned_var(self, fine_var_map, fine_rho):
        """Exact variance of the 2x2 block-sum: sum of the intra-block
        covariance table with the local sigma map."""
        L = (fine_rho.shape[0] - 1) // 2
        sig = np.sqrt(fine_var_map)
        n_b = N_FINE // 2
        var_b = np.zeros((n_b, n_b))
        for oy in (0, 1):
            for ox in (0, 1):
                for oy2 in (0, 1):
                    for ox2 in (0, 1):
                        r = fine_rho[L + oy - oy2, L + ox - ox2]
                        var_b += r * sig[oy::2, ox::2] * sig[oy2::2, ox2::2]
        return var_b

    def binned_rho(self, fine_rho):
        cov_b = binned_kernel_from_fine(fine_rho)
        return cov_b / cov_b[(cov_b.shape[0] - 1) // 2,
                             (cov_b.shape[0] - 1) // 2]


def add_noise_native(model_nat, rng):
    """gu-2022 add_noise recipe per native exposure (G=1)."""
    err = np.sqrt(SIGMA_BKG ** 2 + np.clip(model_nat, 0.0, None) / EXP_TIME)
    return model_nat + rng.normal(0.0, 1.0, size=model_nat.shape) * err, err


def effective_fine_psf(psf, pipe: DrizzleMockPipeline, half=12):
    """The PSF pushed noiselessly through the pixelation+drizzle pipeline.

    The pipeline is periodic-3 shift-variant; the kernel is extracted at the
    phase of the placement pixel (FINE_RENDER//2 global). The render-check
    gate in 05 quantifies the residual phase error.
    """
    canvas = np.zeros((FINE_RENDER, FINE_RENDER))
    c = FINE_RENDER // 2
    k = psf.shape[0] // 2
    canvas[c - k:c + k + 1, c - k:c + k + 1] = psf
    out = pipe(canvas)
    cp = c - CROP0
    eff = out[cp - half:cp + half + 1, cp - half:cp + half + 1].copy()
    eff *= psf.sum() / eff.sum()             # flux-conserving renorm
    return eff
