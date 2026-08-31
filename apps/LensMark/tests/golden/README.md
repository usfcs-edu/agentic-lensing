# Renderer goldens

`sha256.json` pins the exact PNG bytes the canonical renderer (`lensmark/render`, version in
`lensmark/config.py: RENDERER_VERSION`) produces for the nine hand-authored examples in
`examples/nine/deck-NN.lensmark.json` over `examples/nine/deck-NN.png`. `deck-NN.annot.png` are the
reference renders those shas were taken from; they exist only so a mismatch can be *diagnosed*
(`tests/test_render_golden.py::test_golden_sha256` writes `deck-NN.diff.png` - gitignored - and prints
`max |dRGB|` and the percentage of pixels within 8 levels).

What is pinned: the JSON content, `style_defaults.json` / `palette.json`, the vendored fonts in
`lensmark/render/fonts/`, the renderer code, and Pillow (`pillow==12.2.0` in `pyproject.toml` - its
FreeType and resampling kernels decide the bytes; `sha256.json` records the Pillow version used).

Regenerate on purpose only:

    ~/.venvs/lensmark/bin/python -m pytest tests/test_render_golden.py --update-golden -q

then look at a couple of `examples/nine/*.annot.png` (`lensmark render examples/nine`) before committing.
A tiny `max |dRGB|` with ~100 % of pixels within 8 means a resampling/antialiasing change (Pillow bump);
a large one means a geometry or style change.
