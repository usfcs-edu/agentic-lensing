# LensMark

Expert annotation of strong-lens cutouts — coloured arrows with labels, dashed (field galaxy) / dotted (star)
mask circles whose radius is the lens-modelling mask, a fine-dotted Einstein ring with its θ_E label, a legend —
plus **Claude proposals** (model × effort selectable), a **critique workflow** that turns the expert's verdicts on
LLM annotations into data, and exports for few-shot prompting and ML training.

Every image in a campaign directory has three files:

| file | role |
|---|---|
| `<id>.png` | the original cutout — never written |
| `<id>.lensmark.json` | source of truth: geometry in normalized `[u, v]`, sizes in arcsec, provenance, reviews |
| `<id>.annot.png` | the burned-in overlay, regenerated exactly from the JSON (`render.of_json_sha256` pins it) |

plus `<id>.lensmark.log.jsonl` (append-only history), `proposals/` (immutable Claude runs), `critiques/`, `exports/`.
Formats: `API.md` (HTTP contract), `CONTRACT.md` (geometry + DOM contract), `lensmark/schema/*.json`.

## Run

```bash
uv venv ~/.venvs/lensmark --python 3.13 && uv pip install --python ~/.venvs/lensmark/bin/python -e ".[dev]"
export PATH=~/.venvs/lensmark/bin:$PATH
lensmark examples build                      # nine deck tiles -> examples/nine (16" assumed)
lensmark serve examples/nine                 # http://127.0.0.1:8765  (opens the browser)
```

The **Propose** button rides the logged-in `claude` CLI through the Python Agent SDK (no API key). The
model/effort selector maps 1:1 onto `--model` / `--effort`; the app always passes the full model id, pins the PATH
`claude` binary (`LENSMARK_CLAUDE_BIN` to override) and isolates itself from `~/.claude/settings.json`
(`setting_sources=[]`) so your global effort setting never leaks in. `lensmark doctor` shows which binary the SDK
resolved and runs one cheap smoke turn. `LENSMARK_ENGINE=fixture` replaces Claude with canned proposals (all UI QA
runs this way).

CLI mirrors the UI: `lensmark index|render [--check]|propose --id … --model … --effort …|eval|export coco|ds9|masks|fewshot|patch|doctor DIR`.

## Develop / test

```bash
make test                                    # pytest (schema, coords, store, renderer goldens, engines, API, exports) + vitest
make frontend                                # Vite build -> lensmark/static (committed)
make qa-m1 … make qa-m6                      # playwright-cli browser QA per milestone against a throw-away .qa/campaign copy
lensmark render examples/nine --check        # exit 1 if any annotated PNG is stale
```

Renderer determinism: PIL only, vendored DejaVu fonts, 3× supersampling, no PNG metadata, `pillow==12.2.0` pinned —
the golden SHA-256 test (`pytest --update-golden` to re-pin after an intentional style change).

Keyboard: `A` arrow · `G` galaxy mask · `S` star mask · `R` ring · `T` text · `V`/`Esc` select · `⌘S` save · `⌘Z` undo ·
`[` `]` prev/next · `Z` zoom · `1`–`7` verdicts, `Enter` accept, `X` reject (review).
