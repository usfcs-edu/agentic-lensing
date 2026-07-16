"""The worked-example registry.

Its own module for the same reason `registry.py` is: a driver that did
`import worked_examples` while itself being `worked_examples` would register
every example twice, into two different dicts, and silently find none.

It also fixes an asymmetry that existed for as long as there was one guide:

    registry.py          raises ValueError on a duplicate figure slug  (LOUD)
    worked_examples.py   _EX[key] = ...                                (SILENT)

With two guides both numbering chapters from 1, a duplicate `ch05` under the old
code would have been a last-import-wins overwrite, and `verify_numbers.py` would
then have cheerfully validated one book's prose against the other book's numbers.
In a project whose entire premise is "every number reproduces", that is the one
failure mode that actually matters. Physical namespacing (one module per guide,
one import per process) makes it impossible; the guard below makes it loud
anyway, which costs two lines.
"""
from __future__ import annotations

_EX: dict[str, dict] = {}


def example(chapter_key: str, expect: dict | None = None, note: str = ""):
    """Register a worked example.

    ``expect`` = {name: (value, tol)} — values asserted by ``--check``. Use it
    for anything pinned against a number this repository has published; those
    are the guide's credibility.
    """

    def deco(fn):
        if chapter_key in _EX:
            raise ValueError(
                f"duplicate example key: {chapter_key!r} — two examples in one "
                f"process. Each guide has its OWN worked_examples module and "
                f"exactly one is imported per run; see guides.py.")
        _EX[chapter_key] = dict(fn=fn, expect=expect or {}, note=note)
        return fn

    return deco


def run_all() -> dict:
    return {k: {"note": s["note"], "values": s["fn"]()} for k, s in _EX.items()}


def check() -> tuple[int, list[str]]:
    """Assert every pinned value. Returns (n_checked, failures)."""
    fails, n = [], 0
    for key, spec in _EX.items():
        vals = spec["fn"]()
        for name, (want, tol) in spec["expect"].items():
            n += 1
            if name not in vals:
                fails.append(f"{key}.{name}: not returned by the example")
                continue
            got = vals[name]
            if abs(got - want) > tol:
                fails.append(
                    f"{key}.{name}: got {got!r}, expected {want!r} +/- {tol!r}")
    return n, fails
