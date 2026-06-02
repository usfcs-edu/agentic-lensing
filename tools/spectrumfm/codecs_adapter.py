#!/usr/bin/env python
"""
Adapter exposing a trained codecs `Codec` (Mamba3 + Residual-FSQ) through
redshifty's spectrum-tokenizer interface, so it can be swapped in for
`SpectrumTokenizer` in the Approach-A transformer.

redshifty's `tokenize_and_build` calls `spec_tok.encode(x)` and uses element
`[0]` as the (B, T) integer codes, then adds `SPECTRUM_TOKEN_OFFSET` and indexes
the transformer's 1024-code spectrum vocab slot.

RFSQ is a residual hierarchy of FSQ layers (levels [5,4,4] over latent_dim=4):
the compound codebook is 5^4 * 4^4 * 4^4 ~ 40.9M codes — far larger than
redshifty's 1024-code slot. We therefore expose `code_mode`:
  - "layer0"  : the coarse top-level FSQ index only (5^latent_dim = 625 <= 1024)
                — the minimal-blast-radius choice for the redshifty swap.
  - "compound": the full residual index (gate/analysis only; not redshifty-loadable).

INPUT FORMAT. codecs normalizes by per-spectrum median flux:
    good   = ivar > 0
    median = nanmedian(flux[good]);  median[nan] = 1
    flux_norm = flux / median ;  istd_norm = sqrt(ivar) * median
    x_in = stack([flux_norm, istd_norm], dim=1)              # (B, 2, L)
`encode(x)` accepts either redshifty-style x=stack([flux, istd]) (it recovers
ivar=istd^2 and renormalizes) or pre-normalized codecs x via `normalized=True`.

GRID CAVEAT. The spectra must be on the codecs wavelength grid (its HDF5 `wave`).
redshifty-grid spectra need resampling onto the codecs grid before encode();
that resampling is intentionally left to the integration step, not baked here.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

CODECS_REPO = Path("/raid/benson/git/agentic-lensing/lensing-repos/codecs")
if str(CODECS_REPO) not in sys.path:
    sys.path.insert(0, str(CODECS_REPO))

from models.model import Codec  # noqa: E402

# Matches configs/train.yaml model.mamba.
DEFAULT_MAMBA = dict(d_state=128, expand=2, headdim=64, rope_fraction=1, chunk_size=16)


def fsq_codes_to_indices(fsq, codes):
    """Mixed-radix integer index from one FSQ layer's quantized values.

    Reimplemented here because codecs' own FSQ.codes_to_indices does
    `integers.clamp(0, self.levels.long() - 1)` — clamp(int, Tensor), which
    raises on a vector `levels` (it is never exercised in codecs training).
    Every RFSQ layer uses uniform levels ([L]*latent_dim), so a scalar clamp
    is exact.
    """
    half_l = fsq.half_l  # (latent_dim,)
    integers = ((codes * half_l) + half_l).round().long()
    level = int(fsq.levels[0].item())
    integers = integers.clamp(0, level - 1)
    d = integers.shape[-1]
    bases = torch.tensor([level ** k for k in range(d)],
                         device=integers.device, dtype=torch.long)
    return (integers * bases).sum(-1)


def load_codec(ckpt_path, d_model=512, latent_dim=4, rfsq_levels=(5, 4, 4),
               mamba_kwargs=None, device="cuda"):
    """Build a Codec with the train.yaml hyperparameters and load weights."""
    codec = Codec(rfsq_levels=list(rfsq_levels), d_model=d_model,
                  latent_dim=latent_dim, **(mamba_kwargs or DEFAULT_MAMBA)).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    codec.load_state_dict(sd)
    codec.eval()
    return codec


def load_codec_from_config(ckpt_path, config_path, device="cuda"):
    """Build a Codec using the model hyperparameters from the run's yaml config
    (d_model / latent_dim / rfsq_levels / mamba can differ from train.yaml
    defaults — e.g. the local large run used d_model=256, headdim=32)."""
    import yaml
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    m = cfg["model"]
    return load_codec(ckpt_path, d_model=int(m["d_model"]),
                      latent_dim=int(m["latent_dim"]),
                      rfsq_levels=tuple(m["rfsq_levels"]),
                      mamba_kwargs=dict(m["mamba"]), device=device)


class CodecsTokenizerShim(nn.Module):
    """Wraps a Codec; `.encode(x)` returns (indices (B,T), None) like redshifty's
    SpectrumTokenizer.encode (which returns (indices, denorm))."""

    def __init__(self, codec, code_mode="layer0", amp_dtype=torch.bfloat16):
        super().__init__()
        self.codec = codec
        assert code_mode in ("layer0", "compound")
        self.code_mode = code_mode
        self.amp_dtype = amp_dtype
        self.layer_sizes = [int(l.fsq.codebook_size) for l in codec.rfsq.layers]
        self.codebook_size = (self.layer_sizes[0] if code_mode == "layer0"
                              else int(math.prod(self.layer_sizes)))

    @staticmethod
    def codecs_normalize(flux, ivar):
        """(B, L) flux + ivar -> (B, 2, L) median-normalized codecs input."""
        good = ivar > 0
        flux_valid = flux.masked_fill(~good, torch.nan)
        median = flux_valid.nanmedian(dim=1, keepdim=True).values
        median = median.masked_fill(median.isnan(), 1.0)
        flux_norm = flux / median
        istd_norm = torch.sqrt(ivar.clamp(min=0.0)) * median
        return torch.stack([flux_norm, istd_norm], dim=1)

    def _to_codecs_input(self, x, normalized):
        if normalized:
            return x
        # redshifty x = stack([flux, istd]); recover ivar = istd^2.
        flux, istd = x[:, 0], x[:, 1]
        return self.codecs_normalize(flux, istd ** 2)

    @torch.no_grad()
    def layer_indices(self, x, normalized=False):
        """List of per-RFSQ-layer (B, T) integer code indices."""
        xin = self._to_codecs_input(x, normalized)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype):
            _, codes, _ = self.codec.encode(xin)
        return [fsq_codes_to_indices(layer.fsq, c)
                for layer, c in zip(self.codec.rfsq.layers, codes)]

    @torch.no_grad()
    def encode(self, x, normalized=False):
        """redshifty-compatible: returns (indices (B,T) long, None)."""
        idxs = self.layer_indices(x, normalized)
        if self.code_mode == "layer0":
            out = idxs[0]
        else:
            out = torch.zeros_like(idxs[0])
            base = 1
            for li, sz in zip(idxs, self.layer_sizes):
                out = out + li * base
                base *= sz
        return out.long(), None

    @torch.no_grad()
    def perplexity(self, x, normalized=False):
        """codecs normalized RFSQ perplexity in [0,1] (0=collapsed, 1=uniform)."""
        xin = self._to_codecs_input(x, normalized)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype):
            _, codes, _ = self.codec.encode(xin)
        return self.codec.perplexity(codes)
