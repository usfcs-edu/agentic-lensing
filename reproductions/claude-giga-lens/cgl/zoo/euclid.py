"""T4 zoo stub: Euclid Q1 strong-lens candidate posteriors.

P3 SCOPE -- deliberately not implemented in P2. The registry lists this
target as unavailable; get_target raises with this note so no benchmark cell
can silently claim a Euclid result before the P3 data/PSF/noise-model work
(euclid-q1 reproduction) is gated.
"""
from __future__ import annotations

NOTE = ("euclid_q1 is P3 scope: requires the euclid-q1 cutout/PSF/noise "
        "products and their own parity gates before any posterior is "
        "declared. Not available in P2a; see CAMPAIGN.md decisions.")


def build():
    raise NotImplementedError(NOTE)
