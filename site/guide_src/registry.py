"""The figure registry.

Deliberately its own module. If ``figures.py`` did ``from make_figures import
figure`` while ``make_figures.py`` was running as ``__main__``, Python would
import a SECOND copy of make_figures and the decorator would populate a registry
the running script never sees — the builder silently finds zero figures. Both
sides import this module instead, so there is exactly one registry.
"""
from __future__ import annotations

REGISTRY: dict[str, dict] = {}


def figure(slug: str, chapter: int, caption_hint: str = "", width: str = "90%"):
    """Register a figure builder.

    The builder is called once per scheme as ``fn(scheme)`` and must return
    ``(fig, values)`` where ``values`` is a dict of numbers the prose may quote.
    """

    def deco(fn):
        if slug in REGISTRY:
            raise ValueError(f"duplicate figure slug: {slug}")
        REGISTRY[slug] = dict(
            fn=fn, chapter=chapter, caption_hint=caption_hint, width=width
        )
        return fn

    return deco
