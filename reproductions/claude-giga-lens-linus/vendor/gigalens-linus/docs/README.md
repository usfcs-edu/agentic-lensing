# Building the gigalens docs

The documentation is **Sphinx + MyST** (Markdown authoring), organised on the
[Diátaxis](https://diataxis.fr) model (tutorials / how-to / explanation /
reference). The API reference is generated from the source docstrings by autodoc.

## Local build

```bash
python -m venv .venv-docs && source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt   # docs toolchain + light CPU deps
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

The package source is read from `../src` (via `sys.path` in `conf.py`); you do
**not** need to `pip install` gigalens itself, and the build needs **no GPU**.

Live-reload while writing:

```bash
python -m pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html --open-browser
```

## Why the build needs no GPU

autodoc imports each module to read its docstrings. `conf.py` installs the
**light, CPU-only** deps that actually execute at import time (`numpy`, `scipy`,
`jax[cpu]`) and **mocks** only the heavy or optional ones (`tensorflow`,
`tensorflow_probability`, `optax`, `objax`, `lenstronomy`, `blackjax`,
`fastprogress`, …) via `autodoc_mock_imports`. CPU jax imports fine, so no GPU is
needed; the heavy mocks keep CI cheap. (jax must NOT be mocked — several profiles
evaluate constants like `jnp.pi` at import.)

### If autodoc fights the mocks

If you mock a dependency that a class *body* actually calls at definition time,
the import fails. Either install that dep (if light) or, for a page that errors,
switch it to the **static-analysis** backend, which never imports the module:

1. `pip install sphinx-autodoc2`, uncomment it in `docs/requirements.txt`.
2. Add `"autodoc2"` to `extensions` in `conf.py` and configure
   `autodoc2_packages = ["../src/gigalens"]` with a MyST render target.
3. autodoc2 reads the RST-flavoured docstrings statically — no import, no GPU.

This was evaluated as the primary backend; standard autodoc was chosen for
familiarity, with autodoc2 kept as the documented fallback.

## Tutorials come from the demo notebooks

The tutorial pages **are** the demo notebooks — there is no separate transcription
to keep in sync. Each `docs/tutorials/<name>.ipynb` is a symlink to a
`demos/*.ipynb`, and `myst-nb` renders it:

```
docs/tutorials/first-fit.ipynb  ->  ../../demos/simple_demo.ipynb
docs/tutorials/multiband.ipynb  ->  ../../demos/multiband_demo.ipynb
docs/tutorials/cosmology.ipynb  ->  ../../demos/cosmology_demo.ipynb
docs/tutorials/shapelets.ipynb  ->  ../../demos/shapelets_demo.ipynb
docs/tutorials/point-source.ipynb -> ../../demos/lensed_point_source_demo.ipynb
```

Edit a demo → rebuild → the tutorial updates. The narrative (intros, section
prose, admonitions) lives in the notebooks' **markdown cells**, so keep those in
good shape as you edit the demos.

**Cell tags** (myst-nb): tag any code cell in a demo to control how it renders as
a tutorial —

- `remove-cell` — drop the cell entirely (e.g. `%matplotlib inline`, env setup).
- `hide-input` — collapse the code but keep its output (e.g. pure plotting cells).
- `remove-input` / `remove-output` — drop just the code or just the output.

**Figures / outputs.** The docs build **never executes** notebooks
(`nb_execution_mode = "off"` — no GPU in CI). Figures appear only if the notebook
is committed *with outputs*: run the demo on a GPU and commit the executed
notebook. A demo with no outputs renders as prose + code (still a valid tutorial).

**Adding a tutorial.** Symlink the notebook into `docs/tutorials/` under its page
name and add that name to the `tutorials/index.md` toctree.

## Keeping examples from rotting

The long-term goal is that every documented code example is machine-checked, so
the docs cannot silently drift from the API (the failure mode that made the old
`scene-api.md` wrong). Two complementary options:

- `sphinx.ext.doctest` — add `>>>` doctest blocks and run `sphinx-build -b doctest`.
- A CI job that re-executes the `demos/` notebooks on a GPU runner and fails on
  cell errors.

## Viewing / deployment

**Locally (any time):**

```bash
docs/serve.sh --build     # rebuild, then serve on a free localhost port
docs/serve.sh             # serve the last build
```

It prints `http://127.0.0.1:<port>/`; VS Code Remote offers to forward it.

**While the repo is private:** CI (`.github/workflows/docs.yml`) builds the HTML
on every push/PR and uploads it as a downloadable artifact named
**`gigalens-docs-html`**. Open the workflow run → *Artifacts* → download, unzip,
open `index.html`. No hosting, no Enterprise account.

**Going public (e.g. at paper submission):** GitHub Pages deployment is wired but
**disabled** until three things are true — (1) the repo/site is public, (2)
*Settings → Pages → Source = "GitHub Actions"*, (3) repo variable
`PUBLISH_PAGES = true`. Then pushes to `linusu-dev-merge` publish to
`https://seanxuseanxu.github.io/gigalens/`. Versioned docs (a `mike`-style version
switcher) can be added once the API is tagged.

## Structure

```
docs/
  conf.py            build config (extensions, theme, mocks, intersphinx)
  requirements.txt   docs build deps
  index.md           landing + quickstart
  tutorials/         learning-oriented (from demos/)
  how-to/            task-oriented recipes
  explanation/       concepts / design
  reference/         autodoc-generated API reference
```
