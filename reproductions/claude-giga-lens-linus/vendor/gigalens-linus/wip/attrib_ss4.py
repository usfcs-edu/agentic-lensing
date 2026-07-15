#!/usr/bin/env python3
"""Attribution follow-up (pre-registered amendment, 2026-07-09): the direct-chi2
anchor falsifier fired (<5%); the combined big-cell gain at (200,ss4,nmax30) was
-27%. Measure the two missing 2x2 variants there to attribute it: if rfft+image
matches rfft+direct, ALL the gain is rfft2 and the direct-chi2 wiring adds
nothing at scale either."""
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_fast_lstsq import (_time_variant, complex_conv, image_path_fns,
                                 direct_fns)

for label, ctx, fns in [("rfft+image     ", contextlib.nullcontext, image_path_fns),
                        ("complex+direct ", complex_conv, direct_fns)]:
    r = _time_variant(200, 4, 30, ctx, fns, n_iter=30)
    print(f"(200,ss4,nmax30) {label}: grad {r['grad_ms']:8.2f} ms "
          f"(min {r['grad_ms_min']:.2f})  peak {r['xla_peak_mb']:8.0f} MB")
