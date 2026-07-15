#!/usr/bin/env python3
"""Final-configuration timing: ss2 cells (fold auto-off via _FUSE_CONV_POOL_MIN_SS)
new-vs-old, plus an ss4 fused smoke. Part of the c15 checkpoint record."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_fold_stack import time_variant, unfused, buffer_phi  # noqa: E402


def show(cell, label, r):
    npx, ss, nm = cell
    print(f"({npx},ss{ss},nmax{nm}) {label}: grad {r['grad_ms']:7.2f} ms "
          f"(min {r['grad_ms_min']:.2f}) peak {r['xla_peak_mb']:6.0f} MB")


for cell in [(200, 2, 15), (200, 2, 30)]:
    show(cell, "final", time_variant([], *cell, 30))
    show(cell, "old  ", time_variant([unfused, buffer_phi], *cell, 30))
show((200, 4, 30), "final", time_variant([], 200, 4, 30, 10))
