import functools
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax
from lenstronomy.Util.kernel_util import subgrid_kernel
from objax.constants import ConvPadding
from objax.functional import average_pool_2d

import gigalens.model
import gigalens.simulator


def _regularize_gram(gram, jitter_scale=1e-6):
    gram = 0.5 * (gram + jnp.swapaxes(gram, -1, -2))
    diag_mean = jnp.mean(jnp.diagonal(gram, axis1=-2, axis2=-1), axis=-1, keepdims=True)
    jitter = jitter_scale * jnp.maximum(diag_mean, 1.0)
    return gram + jitter[..., jnp.newaxis] * jnp.eye(gram.shape[-1], dtype=gram.dtype)


def _triangular_solve_from_cholesky(chol, rhs):
    y = lax.linalg.triangular_solve(chol, rhs, left_side=True, lower=True)
    return lax.linalg.triangular_solve(
        jnp.swapaxes(chol, -1, -2), y, left_side=True, lower=False
    )


def _solve_normal_eq_with_fallback(gram, rhs):
    def solve_one(gram_i, rhs_i):
        gram_i = _regularize_gram(gram_i)
        chol = jnp.linalg.cholesky(gram_i)
        chol_solution = _triangular_solve_from_cholesky(chol, rhs_i)
        return lax.cond(
            jnp.any(jnp.isnan(chol_solution)),
            lambda: jnp.linalg.lstsq(gram_i, rhs_i)[0],
            lambda: chol_solution,
        )

    if gram.shape[0] == 1:
        return solve_one(gram[0], rhs[0])[jnp.newaxis, ...]

    return jax.vmap(solve_one)(gram, rhs)


def _shared_kernel_component_conv(img, flat_kernel):
    bs, depth, height, width = img.shape
    folded = jnp.reshape(img, (bs * depth, 1, height, width))
    convolved = lax.conv(folded, flat_kernel, (1, 1), "SAME")
    return jnp.reshape(convolved, (bs, depth, height, width))


def _weighted_lstsq_reconstruct(ret, observed_image, err_map):
    with jax.named_scope("linear_solve"):
        W = (1 / err_map)[..., jnp.newaxis]
        Y = jnp.reshape(observed_image * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((ret * W), (ret.shape[0], -1, ret.shape[-1]))
        Xt = jnp.transpose(X, (0, 2, 1))
        coeffs = _solve_normal_eq_with_fallback(Xt @ X, Xt @ Y)[..., 0]

    with jax.named_scope("recombine_components"):
        recon = jnp.sum(ret * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)
    return recon, coeffs


class LensSimulator(gigalens.simulator.LensSimulatorInterface):
    def __init__(
            self,
            phys_model: gigalens.model.PhysicalModel,
            sim_config: gigalens.simulator.SimulatorConfig,
            bs: int,
    ):
        super(LensSimulator, self).__init__(phys_model, sim_config, bs)
        self.supersample = int(sim_config.supersample)
        self.transform_pix2angle = (
            jnp.eye(2) * sim_config.delta_pix
            if sim_config.transform_pix2angle is None
            else sim_config.transform_pix2angle
        )
        self.conversion_factor = jnp.linalg.det(self.transform_pix2angle)
        self.transform_pix2angle = self.transform_pix2angle / float(self.supersample)
        _, _, img_X, img_Y = self.get_coords(
            self.supersample, sim_config.num_pix, np.array(self.transform_pix2angle)
        )
        self.img_X = jnp.repeat(img_X[..., jnp.newaxis], bs, axis=-1)
        self.img_Y = jnp.repeat(img_Y[..., jnp.newaxis], bs, axis=-1)

        self.numPix = sim_config.num_pix
        self.bs = bs
        self.depth = sum([x.depth for x in self.phys_model.lens_light]) + sum(
            [x.depth for x in self.phys_model.source_light])
        self.kernel = None
        self.flat_kernel = None

        if sim_config.kernel is not None:
            kernel = subgrid_kernel(
                sim_config.kernel, sim_config.supersample, odd=True
            )[::-1, ::-1, jnp.newaxis, jnp.newaxis]
            self.kernel = jnp.repeat(kernel, self.depth, axis=2)
            self.flat_kernel = jnp.transpose(kernel, (2, 3, 0, 1))

    @functools.partial(jit, static_argnums=(0,))
    def _beta(self, lens_params: List[Dict]):
        with jax.named_scope("beta_deflection"):
            beta_x, beta_y = self.img_X, self.img_Y
            for lens, p in zip(self.phys_model.lenses, lens_params):
                f_xi, f_yi = lens.deriv(self.img_X, self.img_Y, **p)
                beta_x, beta_y = beta_x - f_xi, beta_y - f_yi
            return beta_x, beta_y

    @functools.partial(jit, static_argnums=(0,))
    def simulate(self, params, no_deflection=False):
        with jax.named_scope("simulate"):
            lens_params = params[0]
            lens_light_params, source_light_params = [], []
            if len(self.phys_model.lens_light) > 0:
                lens_light_params, source_light_params = params[1], params[2]
            else:
                source_light_params = params[1]
            beta_x, beta_y = self._beta(lens_params)
            if no_deflection:
                beta_x, beta_y = self.img_X, self.img_Y
            with jax.named_scope("render_light"):
                img = jnp.zeros_like(self.img_X)
                for lightModel, p in zip(self.phys_model.lens_light, lens_light_params):
                    img += lightModel.light(self.img_X, self.img_Y, **p)
                for lightModel, p in zip(self.phys_model.source_light, source_light_params):
                    img += lightModel.light(beta_x, beta_y, **p)
            img = jnp.transpose(img, (2, 0, 1))
            with jax.named_scope("conv_pool"):
                ret = (
                    lax.conv(img[:, jnp.newaxis, ...], self.flat_kernel, (1, 1), "SAME")
                    if self.flat_kernel is not None
                    else img
                )
                ret = (
                    average_pool_2d(ret, size=self.supersample, padding=ConvPadding.SAME)
                    if self.supersample != 1
                    else ret
                )
            return jnp.squeeze(ret) * self.conversion_factor

    @functools.partial(jit, static_argnums=(0,4,))
    def lstsq_simulate(
            self,
            params,
            observed_image,
            err_map,
            no_deflection=False,
    ):
        with jax.named_scope("lstsq_simulate"):
            lens_params = params[0]
            lens_light_params, source_light_params = [], []
            if len(self.phys_model.lens_light) > 0:
                lens_light_params, source_light_params = params[1], params[2]
            else:
                source_light_params = params[1]
            beta_x, beta_y = self._beta(lens_params)
            if no_deflection:
                beta_x, beta_y = self.img_X, self.img_Y
            with jax.named_scope("build_basis"):
                img = jnp.zeros((0, *self.img_X.shape))
                for lightModel, p in zip(self.phys_model.lens_light, lens_light_params):
                    img = jnp.concatenate((img, lightModel.light(self.img_X, self.img_Y, **p)), axis=0)
                for lightModel, p in zip(self.phys_model.source_light, source_light_params):
                    img = jnp.concatenate((img, lightModel.light(beta_x, beta_y, **p)), axis=0)
                img = jnp.transpose(img, (3, 0, 1, 2))  # bs, n components, h, w
            with jax.named_scope("conv_pool"):
                ret = (
                    _shared_kernel_component_conv(img, self.flat_kernel)
                    if self.flat_kernel is not None
                    else img
                )
                ret = average_pool_2d(ret, size=(self.supersample, self.supersample),
                                      padding="SAME") if self.supersample != 1 else ret
                ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n components

            ret, coeffs = _weighted_lstsq_reconstruct(ret, observed_image, err_map)
            return jnp.squeeze(ret), jnp.squeeze(coeffs)

