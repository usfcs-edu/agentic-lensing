# LensMark — lens-annotation app (plan)

## Context

Xiaosheng's deck (`apps/LensMark/examples/Xiaosheng-Claude-Fable5-Lens-Annotations.pptx`) shows the
target product: a 16″ cutout with **coloured arrows + labels**, a per-panel **legend**, **mask circles**
(dashed = field galaxy, dotted = star; radius = the mask to apply in lens modelling), a **fine-dotted
Einstein ring** with a `θ_E ≈ 1.5″` label, and a free-text description that refers to arrows by colour.
Today those overlays exist only as one-off Claude sessions. LensMark makes them a durable, reproducible
dataset: a directory of cutouts where every image has **(1) the original, (2) the annotated image,
(3) a JSON file** from which (2) is regenerated exactly; a human expert annotates quickly; a button asks
Claude Code (selectable **model × effort**) for a proposal the human accepts/edits/rejects, producing
**critique data** on LLM annotations; accepted items export as **few-shot bundles and ML labels**; optional
**voice** commands. From the 2026‑08‑12 meeting: negative annotations ("spiral arm — NOT an arc") are a
first-class use (few-shot on half the systems, apply to the other half).

**Decisions taken with Greg (2026‑08‑30):** stack = **local web app (Python backend + browser)**;
v1 primitives = **deck vocabulary only** (`arc`/`polygon` reserved for v1.1); **no embargo** on this new
corpus; v1 runs on **Greg's Mac** with the logged-in `claude` CLI (reviewer identity is a field, not auth).

### Why this stack (research summary)
| | Local web app (chosen) | Electron (+Python sidecar / TS-only) | Tauri 2 | Native SwiftUI |
|---|---|---|---|---|
| Reuse of existing tested code | `golden/annotate.py` primitives, `parse.py`, `hooks.py`, Agent-SDK patterns, pandas eval joins | shell only / none (TS rewrite, two renderers drift) | none | none (CoreGraphics rewrite) |
| "annot.png regenerable from JSON" | one server-side PIL rasteriser, byte-stable goldens | Skia goldens tied to engine+platform | same | CG |
| playwright-cli 0.1.13 | native (`open http://127.0.0.1:8765`, DOM snapshot of SVG) | web-mode only; shell via `attach --cdp` | none on macOS (no WKWebView driver) | none (XCUITest needs Xcode; **no Xcode installed**) |
| Distribution | `uv run lensmark DIR`; later ssh/Tailscale tunnel for a colleague | unsigned + `xattr`, or $99/yr Developer ID + notarize; colleague still needs `claude` logged in | same wall | hand-built bundle, ad-hoc sign |
| Voice | Safari/Chrome Web Speech + server STT tier | Web Speech dead in Electron | works, flaky permissions | best (SpeechAnalyzer) — extractable as a helper for any stack |

A desktop wrap (Electron with the Python server as sidecar, or `pywebview`) stays a one-milestone add-on
because the UI talks to the core only over the HTTP contract below. The **web-based variant** is the same
server bound beyond localhost behind an ssh/Tailscale tunnel (Claude calls ride Greg's login; reviewer is
a field; per-id write lock) — no separate codebase.

## Architecture

```
browser (Vite + vanilla TS, SVG overlay = live preview)  ──HTTP/SSE──►  lensmark (FastAPI/uvicorn, 127.0.0.1:8765)
                                                                          ├─ store.py    dir scan, atomic JSON writes, append-only log, manifest
                                                                          ├─ render.py   PIL canonical renderer  →  <id>.annot.png
                                                                          ├─ claude/     Agent SDK engine (model×effort), fixture engine, fake CLI shim
                                                                          ├─ critique.py / evaluate.py / exports/ / voice/
                                                                          └─ cli.py      serve | index | render | propose | eval | export | doctor
```
Python package `lensmark` in a **new uv venv `~/.venvs/lensmark`** (py3.13; pins `pillow==12.2.0` for golden
images, `claude-agent-sdk>=0.2.148`, `fastapi`, `uvicorn`, `pydantic>=2`, `numpy`, `pandas`). Front end
built with `npm run build` into `lensmark/static/` (committed) so runtime needs no node.

## Directory layout — `apps/LensMark/`

```
pyproject.toml  Makefile (venv|frontend|test|qa-mN|check-render)  README.md  .gitignore (.playwright-cli/, node_modules/)
schema/            lensmark-1.0.schema.json  lensmark-proposal-1.0.schema.json  lensmark-patch-1.0.schema.json
                   lensmark-critique-1.0.schema.json  palette.json  style_defaults.json   (served at /api/schema/*, /api/style)
lensmark/
  cli.py config.py model.py coords.py palette.py store.py validate.py
  render/ primitives.py draw.py fonts/DejaVuSans-Bold.ttf
  claude/ engine.py propose.py parse.py trace.py mcp_tools.py prompts/{propose_v1.md,patch_v1.md}
  server/ app.py api_images.py api_annotations.py api_propose.py api_critique.py api_export.py api_voice.py sse.py
  critique.py evaluate.py exports/{fewshot,coco,ds9,masks}.py voice/{patch,stt}.py
  static/            committed Vite build
frontend/  package.json vite.config.ts src/{main,api,state,coords,svg/overlay,svg/tools,ui/*}.ts src/coords.test.ts
tests/     test_{schema,coords,store,render_golden,validate,engine,fake_claude_cli,propose,critique_eval,export,api,patch}.py
           golden/*.lensmark.json + sha256.json   fixtures/{fake_claude, proposals/*.json, coords_cases.json}
           qa/m1_manual.sh … m6_voice.sh   (playwright-cli scripts; run from apps/LensMark)
examples/  <deck pptx>   nine/ (built by `lensmark examples build`: deck-01..09.png + lensmark.config.json)
```
Never name a directory `data` (root `.gitignore:36` is a bare `data` rule). Campaign dirs (user data) live
outside the repo.

## The three files + campaign directory

```
<campaign>/
  lensmark.config.json           campaign defaults: cutout_arcsec (16.0), survey, north_up/east_left, array_origin,
                                 default model/effort, optional per-image overrides table {id: {cutout_arcsec, ra, dec}}
  lensmark.manifest.json         DERIVED (rebuilt by `lensmark index`): id, sha, status, n_items, grade, annot_stale
  <id>.png                       (1) original — never written
  <id>.annot.png                 (2) burned-in overlay — written on every save and by `lensmark render`; `--check` detects staleness
  <id>.lensmark.json             (3) source of truth
  <id>.lensmark.log.jsonl        append-only ops {ts, actor, op, item_id, before, after, source: ui|voice|claude|cli}
  proposals/<id>.<run_id>.json   immutable Claude runs (request, prompt sha, raw structured_output, validated items, usage, cost)
  critiques/<id>.<reviewer>.<run_id>.json
  exports/{fewshot,coco,ds9,masks,eval}/
```
`<id>` = image file stem; accepted stems: `J<ra><dec>`, `rank-NNN`, or anything `[A-Za-z0-9_.-]+`.

## JSON format v1 (`lensmark/1.0`)

Coordinates: geometry in **normalized `[u, v]`** (0..1, origin top-left, u→right, v→down); all physical
sizes in **arcsec** (`radius_arcsec`, `theta_e_arcsec`); never store pixels. `image` carries `width, height,
cutout_arcsec, pixel_scale_arcsec (= cutout/width, asserted), array_origin ("upper" — DESI PNGs are y-flipped
by `common/render.py:55`), north_up, east_left, sha256, survey, wcs?`. Conversions (in `coords.py`, mirrored in
`frontend/src/coords.ts`, one shared fixture file): `px=(u·W, v·H)`; `dE=(0.5−u)·cutout` if east_left,
`dN=(0.5−v)·cutout`; `(r, PA_E_of_N)=(hypot, atan2(dE,dN))` — same anchors as `annotate.py:11-21`;
DS9 image coords `x=u·W+0.5, y=(1−v)·H+0.5`.

Palette is a closed set of names (prose says "the magenta arrow"): `magenta cyan green yellow white orange
gray` for arrows (deflector = green by convention), `mask_red` for masks only, `ring_white` for the ring.
`style_defaults` (fractions of min(W,H); `gap_mult: 3.0` encodes deck PROMPT 6 "gap = 3× dot radius") live in
the file so the PIL renderer and the SVG preview read identical constants.

```json
{ "schema_version": "lensmark/1.0", "id": "deck-01", "created": "…", "modified": "…",
  "image": { "file": "deck-01.png", "sha256": "…", "width": 410, "height": 410, "cutout_arcsec": 16.0,
             "pixel_scale_arcsec": 0.03902, "array_origin": "upper", "north_up": true, "east_left": true,
             "survey": null, "wcs": null },
  "system": { "object_id": null, "rank": 91, "grade": null, "score_1_4": null, "confidence_lmh": null,
              "theta_e": { "value_arcsec": 1.5, "method": "geometric", "alt_arcsec": null },
              "verdict": "likely_lens", "description": "…the cyan arrow marks a tight arc…", "tags": [] },
  "legend": { "show": true, "position": "auto" },
  "items": [
    { "id": "ann-arrow-001", "type": "arrow", "tail": [0.41,0.70], "head": [0.446,0.586], "color": "cyan",
      "label": "tight arc", "label_anchor": "auto", "show_in_legend": true,
      "created_by": { "kind": "human", "reviewer": "xhuang" }, "created_at": "…", "status": "accepted", "review": null },
    { "id": "ann-mask-001", "type": "mask_circle", "center": [0.043,0.352], "radius_arcsec": 1.28, "kind": "galaxy",
      "color": "mask_red", "created_by": { "kind": "claude", "model": "claude-opus-5", "effort": "xhigh", "run_id": "run-…" },
      "status": "proposed", "review": null },
    { "id": "ann-ring-001", "type": "einstein_ring", "center": [0.514,0.494], "theta_e_arcsec": 1.5,
      "center_ref": "ann-arrow-002", "label": "θ_E ≈ 1.5″", "color": "ring_white", "status": "accepted" },
    { "id": "ann-text-001", "type": "text", "pos": [0.06,0.94], "text": "seeing 1.1″", "color": "white", "status": "accepted" } ],
  "provenance": { "proposal_runs": [ { "run_id", "model", "effort", "engine", "prompt_sha256", "fewshot_sha256",
                                       "cost_usd", "duration_s", "n_items_proposed", "n_invalid", "n_repaired", "proposal_file" } ],
                  "critiques": [], "log": "deck-01.lensmark.log.jsonl" },
  "render": { "renderer": "lensmark-render/0.1.0", "output": "deck-01.annot.png", "of_json_sha256": "…" } }
```
Rules (pydantic v2, `extra="forbid"` like `golden/schemas_panel.py:61`): item `type` discriminated union
`arrow | mask_circle | einstein_ring | text` (`arc`, `polygon` reserved); `mask_circle.kind` drives stroke
(galaxy→dash, star→dot); `status ∈ proposed|accepted|edited|rejected|invalid` (`edited` keeps `edit_of`);
`review.verdict ∈ correct|wrong_position|wrong_label|wrong_type|wrong_size|spurious|redundant|missed_by_model`
(`missed_by_model` on human-added items = the recall signal); `system.grade` A–D (`common/schemas.py:15`)
plus optional `score_1_4`/`confidence_lmh` (golden campaign scale); validator warns on colour words in
`description` with no matching item. History = current file + append-only log (no `revisions[]`).

## Deterministic renderer (`lensmark/render/`)
- PIL only; 3× supersampled RGBA overlay → `LANCZOS` down → alpha-composite on the original; PNG with
  `optimize=False`, no metadata. Output size = `image.width × height`. Vendored `DejaVuSans-Bold.ttf`
  (from `~/.venvs/lensjudge/.../matplotlib/mpl-data/fonts/ttf/`) loaded by path — never system fonts
  (`annotate.py:104,251-268` fallback chain is the drift hole).
- Vendor (copy with `# from reproductions/lensjudge/golden/annotate.py:Lnn` headers): `dashed_polyline`
  (:352), `_text`/`_text_size` halo text (:281/:273), `_place_label`/`_clamp_label` collision avoidance
  (:303/:339, generalised to an arbitrary centre and the whole image), `wrap_parts` legend wrapping (:554);
  the labels-below-geometry layering rule (:1-56). Port from `~/sync/research/jwst-strong-lens-search/scripts/21_annotate.py`
  (read-only, never write there): `arrow()` (:54, tip stops `gap` short, 28° barbs → filled head),
  `dashed_circle`/`dotted_circle` (:68/:76 → dots placed by arc length, phase at angle 0, `gap = gap_mult·dot_r`),
  `approach_angle()` (:84, arrow comes from outside so the shaft never crosses the feature; used for
  `label_anchor:auto` and for Claude-proposed arrows that give only `head`), legend plate (:152-165).
- Draw order: masks → ring → arrows → text → labels → legend → θ_E label; labels clamped inside the image
  (fixes the deck's clipped "arc segment (SE)").
- Tests: 9 hand-authored golden JSONs → `sha256(png) == tests/golden/sha256.json` (`--update-golden`;
  on failure print `ImageChops` max ΔRGB / % pixels so a Pillow bump is diagnosed); render-twice byte
  identity; scale invariance (render at 410 vs 820 px, downsample, mean |Δ| < 6); colour-mask asserts reusing
  `tests/test_golden_annotate.py:81-103` helpers (arrow-head centroid within 1.5 px of `head·W`, dot count on
  the ring within ±10 %, galaxy vs star stroke differ, label bbox inside canvas); `lensmark render --check`.

## Claude proposal (`lensmark/claude/`) — model × effort
Engine = Python `claude_agent_sdk.query()` (rides the logged-in CLI; no API key), port of
`imaging/grader_lean.py:64-90 _collect` + `:139-151` options with these **fixes**:
```python
ClaudeAgentOptions(
    model=FULL_ID[model],                      # claude-fable-5 | claude-opus-5 | claude-sonnet-5 | claude-haiku-4-5 (never aliases)
    effort=effort,                             # low|medium|high|xhigh|max → --effort ; omitted/greyed for haiku
    thinking={"type": "adaptive", "display": "summarized"},
    output_format={"type": "json_schema", "schema": PROPOSAL_SCHEMA},   # flat, additionalProperties:false → --json-schema
    system_prompt=PROPOSE_SYSTEM, setting_sources=[],                    # [] emits --setting-sources= ; None would load ~/.claude effortLevel:xhigh
    tools=[], permission_mode="bypassPermissions", max_turns=2, max_budget_usd=budget,
    include_partial_messages=True, cli_path=os.environ.get("LENSMARK_CLAUDE_BIN") or shutil.which("claude"),
    hooks=trace.hooks(), cwd=str(campaign_dir))
```
`cli_path` is pinned to the PATH `claude` (2.1.251) because the SDK's `_find_cli`
(`claude_agent_sdk/_internal/transport/subprocess_cli.py:81-86`) prefers its **bundled** binary (2.1.172 in
0.2.96) which lacks the Claude‑5 model ids. Image input = streaming-input user message with content blocks
`[{"type":"text", meta}, {"type":"image","source":{"type":"base64",…}} ×2]`: the cutout upsampled to ≥400 px
(like `config.RENDER_PX`) and a **grid overlay** copy (0.1-step u/v ticks) so the model names normalized
coordinates. M3.5 option: MCP `get_cutout_image(views=[full,zoom,grid])` (copy `tools/cutout.py:56-79` +
`tools/server.py:54-72`). Prompt (`prompts/propose_v1.md`, sha recorded per run): coordinate contract, item
types, palette rules (deflector = green; refer to arrows by colour), label vocabulary from the deck, mask
semantics, "cap masks at 12, do not circle diffraction spikes", θ_E from geometry with `method`/`alt`,
negative labels welcome, and the **self-verify step** from `annotate_workflow.js:116-121`. Few-shot block =
K (original, annotated, JSON, description) tuples, stable prefix first for prompt caching.

Post-processing: re-validate `ResultMessage.structured_output` → fallback `parse.extract_json_block`
(vendor `common/parse.py:18-77`; its inside-brace quote tracking exists for `1.5"` labels) → one repair
turn (`grader_lean.py:29-34,159-172`). Lenient repair table (spirit of `common/schemas.py:36-39`): clamp
uv in [−0.05, 1.05], else `status:invalid/out_of_bounds`; `radius ≤ 0 or > cutout/2` → `bad_radius`;
`|tail−head| < 0.01` → `degenerate_arrow`; unknown colour → nearest palette; θ_E > cutout/2 → invalid.
Counters `n_invalid/n_repaired` are dataset columns. App mints ids, stamps `created_by` from the
**request** (model, effort, run_id), `status:"proposed"`, writes `proposals/<id>.<run_id>.json` + trace
(`common/hooks.py:42-88`, images elided), appends `provenance.proposal_runs`.

Server: `POST /api/propose/{id} {model, effort, budget, fewshot?}` → 202 `{run_id}`;
`GET /api/propose/{id}/{run_id}/events` SSE `queued|started|thinking|partial|tool|validated|done|error`
(15 s keepalive; Fable turns can take minutes); `POST …/cancel`. UI: model×effort selects, running cost
from `total_cost_usd`, proposals as a **ghost layer** (dashed, 50 % alpha) until reviewed.

Test doubles (three levels): `LENSMARK_ENGINE=fixture` (returns `tests/fixtures/proposals/<id>.json` with
synthetic SSE — used by all playwright QA); `MockTransport` passed to `query(transport=…)` (exercises real
`_collect`/validate in pytest); executable `tests/fixtures/fake_claude` speaking the stream-json handshake
(`_internal/query.py` initialize control_request → success; then `system/init`, `assistant`, `result` with
`structured_output`), recording argv so a test asserts `--model claude-opus-5 --effort xhigh --json-schema …
--setting-sources=` were sent. `lensmark doctor`: prints resolved binary + one `sonnet/low` turn under
`max_budget_usd 0.10` with cost.

## Critique + evaluation
Per proposed item: keys `1..7` = verdict enum, `Enter` accept, `X` reject, drag = edit (`edit_of`,
`delta_arcsec`); human additions during review → `missed_by_model`; panel scores (completeness /
geometry / labels / description 1–5, `theta_e_human_arcsec`, `would_use_as_fewshot`, free text);
`lead_time_s`. Submit → `critiques/<id>.<reviewer>.<run_id>.json` + statuses in the main JSON + log + re-render.
`lensmark eval --by model,effort` → `exports/eval/{items.parquet, runs.csv}`: precision, recall, median
|Δ| arcsec, θ_E error, spurious-mask rate, parse_ok, cost/image — columns shaped like
`golden/regrade_scrambled.py` FROZEN_COLS (model, effort, thinking) so it joins lensjudge tables.

## Exports
- **few-shot** `lensmark export fewshot --k K [--stratify grade,theta_e]`: images with no `proposed` items;
  optional `--require-flag` (`would_use_as_fewshot`) as a quality filter (no embargo per Greg); content-addressed
  bundle `exports/fewshot/{manifest.json, NNN-<id>.{png,annot.png,lensmark.json,md}, prompt.sha256}`;
  `propose --fewshot exports/fewshot` records `fewshot_sha256`.
- **COCO**: arrows → 2 keypoints + skeleton; masks → 64-gon + `attributes{radius_arcsec,kind}`; ring → bbox +
  attrs; only `accepted|edited`. **DS9 `.reg`** (hand-rolled; `regions` not installed): `image` 1-based y-up or
  `fk5` if `wcs`; galaxy `dash=1 dashlist=8 3`, star `dashlist=2 6`, ring `dashlist=1 3`, arrows `line … # line=0 1`,
  `tag={id:…}`. **masks**: `<id>.mask.png` union of accepted masks at native scale
  (`r_px = r_arcsec / native_pixel_scale_arcsec`).

## Voice (M6, optional)
Text is the contract: `POST /api/patch/{id} {transcript}` → Claude at `effort:low` with the patch schema
(inputs: current JSON, image, transcript, coordinate vocabulary: clock positions, quadrants, compass via
`north_up/east_left`, "a bit bigger" = ×1.25) → id-addressed ops `add|update|delete` (`$system` reserved,
`confidence`, `rationale`, `clarification`) → ghost overlay → per-op ✓/✗ → applied ops logged `source:"voice"`.
STT tier 1 = browser `SpeechRecognition||webkitSpeechRecognition` + a typed transcript box (the tested path);
tier 2 later = `POST /api/stt` → `mlx-whisper` or a ~150-line Swift `SpeechAnalyzer` file transcriber
(verified to typecheck with CLT-only Swift 6.3.3).

## Front end (Vite + vanilla TS, SVG overlay)
Layout: left image list (thumb, status badges, stale-render badge); centre stage `<img>` + `<svg
viewBox="0 0 W H">` in image pixels (DPR/zoom independent; mouse → image px via `getScreenCTM().inverse()`);
right sidebar: item list/editor, system block, Propose panel, Critique panel, Voice box. Keyboard-first
(precedent `golden/tool/grade_template.html:18-19`): `A` arrow (drag tail→head), `G` galaxy mask (drag
centre→radius), `S` star mask, `R` ring (click centre, radius from θ_E field), `T` text, `Esc` select,
`Del`, `⌘Z` (log replay), `⌘S` save, `[`/`]` prev/next, `Z` zoom, `L` legend corner. Autosave draft to
`localStorage` (`lensmark:<dir>:<id>`), `Cache-Control: no-store` (`serve.py:84-91`). Test hooks:
`window.__lensmark = {state, addItem, selectItem, save}`; `data-testid` on tools/inputs.

## Milestones (each ends with `make test` + a playwright-cli QA script; QA fan-out per memory rule: one
Workflow agent per script, own `-s=<name>` session, `resize 1920 1080`, screenshot, **Read** it, `console`, `close`)

**M0 — skeleton + examples (S).** `pyproject.toml`, `Makefile`, `config/model/coords/store`, schema JSON files,
`frontend/` scaffold, `lensmark examples build` (unzip pptx read-only; split `ppt/media/image6.jpg` 1249² on the
dark gutters → `examples/nine/deck-01..09.png`; `lensmark.config.json` with `cutout_arcsec 16.0` flagged
`assumed`, ranks 91,89,68,74,60,41,30,24,40 and θ_E from slides 10–18 as reference). Tests: schema
round-trip + forbid-extras, `coords_cases.json` (uv↔px↔arcsec↔fits, flip once, 4 compass anchors as in
`test_golden_annotate.py:123`), store atomic write + sha mismatch → error. Verify: `lensmark index examples/nine`
→ 9 rows; `lensmark serve examples/nine --engine fixture` → `playwright-cli open http://127.0.0.1:8765`,
`snapshot` lists 9 items, `console` clean.

**M1 — load dir, view, manual annotations, save (L).** `server/{app,api_images,api_annotations,sse}`,
`frontend/src/*`, `tests/test_api.py` (httpx), `tests/qa/m1_manual.sh`:
```
playwright-cli -s=lm open http://127.0.0.1:8765/ ; resize 1920 1080 ; snapshot ; click e<deck-01>
playwright-cli -s=lm eval "document.querySelector('#overlay').getAttribute('viewBox')"          # "0 0 410 410"
playwright-cli -s=lm press a ; mousemove X0 Y0 ; mousedown ; mousemove X1 Y1 ; mouseup             # real drag
playwright-cli -s=lm fill e<label> "tight arc" ; select e<color> cyan ; press Control+s ; requests   # PUT /api/ann/deck-01 200
python -c "…assert items[0]['type']=='arrow' and label=='tight arc' and 0<=head[1]<=1"
playwright-cli -s=lm reload ; eval "document.querySelectorAll('#overlay .annot.arrow').length"    # 1 (from JSON, not localStorage)
playwright-cli -s=lm screenshot --filename=$SCRATCH/m1.png ; console ; close
```
Repeat for galaxy mask (`g`), star mask (`s`), ring (`r`); assert `kind`/`radius_arcsec`/`theta_e_arcsec` in JSON
and that `.mask.galaxy` vs `.mask.star` `stroke-dasharray` differ.

**M2 — deterministic annotated PNG (M).** `render/{primitives,draw}.py`, fonts, `cli render [--check] [--scale]`,
`GET /api/images/{id}/annot`, "Rendered" toggle. Tests above. Verify: `lensmark render examples/nine` writes 9
`.annot.png`; `--check` passes; edit a JSON by hand → `--check` exits 1 → `render` → 0; UI toggle shows
`.annot.png` with `naturalWidth == 410`; screenshot preview vs rendered, Read both.

**M3 — Claude proposal with model/effort + streaming (M/L).** `claude/*`, `server/api_propose.py`,
`ui/propose.ts`, fake CLI, `doctor`. Tests: fake-CLI argv assertion, MockTransport run (N items, 1 invalid,
2 repaired, cost captured), repair table, parse fallback on prose with arcsec marks, SSE order. Verify
(`m3_propose.sh`, fixture engine and once with `LENSMARK_CLAUDE_BIN=tests/fixtures/fake_claude --engine sdk`):
`select e<model> opus ; select e<effort> xhigh ; click e<Propose>` → poll `eval` until `done` → ghost count
== fixture count → `requests` shows `POST /api/propose/... 202` + SSE → `ls proposals/deck-02.*.json` →
`jq .model,.effort` → screenshot. Then one real smoke: `lensmark propose examples/nine --id deck-01 --model
sonnet --effort low --max-budget 0.10`, inspect cost and which binary `doctor` resolved.

**M4 — critique (M).** `critique.py`, `evaluate.py`, `server/api_critique.py`, `ui/critique.ts`. Verify: after
M3 fixture run — `press 1` on ghost 1, `press 6` on 2, drag 3's head, add a human arrow, `click e<Submit>`;
JSON statuses `accepted/rejected/edited/accepted`, new item `missed_by_model`; `ls critiques/`;
`lensmark eval --by model,effort` prints one `opus,xhigh` row.

**M5 — few-shot + ML export (M).** `exports/*`, `server/api_export.py`, `propose --fewshot`. Verify: COCO
JSON asserts (ids unique, bbox in image, categories), `.reg` round-trips ids via the hand parser and uses
1-based y-up, mask PNG pixel count ≈ Σπr² at native scale, few-shot bundle sha stable across two builds;
`lensmark propose --engine fixture --fewshot exports/fewshot` records `fewshot_sha256`.

**M6 — voice (M, optional).** `voice/*`, `server/api_voice.py`, `ui/voice.ts`, `prompts/patch_v1.md`. Verify
(mic-free): `fill e<transcript> "put a dashed circle around the galaxy at upper left" ; click e<Apply voice>`
→ one ghost `mask_circle kind=galaxy` with `u<0.5, v<0.5` (fixture patch) → `press Enter` → JSON has it, log
line `source:"voice"`.

Follow-ons (not in v1): Electron/`pywebview` wrap with the server as sidecar; `lensmark import` from lensjudge
FITS (`common/render.py` 16″ DESI views; `golden/stamps/*.fits` JWST) so real campaigns get exact pixel scales;
Tailscale/ssh serving for Xiaosheng; Swift `SpeechAnalyzer` helper.

## Existing code to reuse (file:line → LensMark home)
- `reproductions/lensjudge/golden/annotate.py:1-56` rules; `:281 _text`, `:273 _text_size`, `:303/:339 _place_label/_clamp_label`,
  `:352 dashed_polyline`, `:554 wrap_parts` → `render/primitives.py`; `tests/test_golden_annotate.py:81-103,115-135` → test helpers/anchors.
- `~/sync/research/jwst-strong-lens-search/scripts/21_annotate.py:54,68,76,84,152-176` (arrow, dashed/dotted circle,
  approach_angle, legend/θ_E plate) and `results/annotations/*.json` (dE/dN + `environment[].radius_arcsec` precedent);
  `scripts/annotate_workflow.js:70-135` (geometry-in-prompt, measure list, self-verify, output JSON) → `prompts/propose_v1.md`. Read-only.
- `reproductions/lensjudge/common/parse.py:18-86` → `claude/parse.py`; `common/hooks.py:42-88` → `claude/trace.py`;
  `imaging/grader_lean.py:29-34,64-90,139-172` → `claude/engine.py`; `config.py:84-91,109-129` model/effort env shape → `config.py`;
  `tools/cutout.py:56-79` + `tools/server.py:54-72` → optional `claude/mcp_tools.py`; `imaging/run_batch.py:122-158` resume+semaphore → `cli propose --all`.
- `golden/tool/serve.py:29-38,84-91,110-118` (contract-key validation, `no-store`, append-only JSONL) and
  `golden/tool/grade_template.html:18-19,151-215` (keyboard UX, localStorage/session model) → `server/api_annotations.py`, `frontend/src/state.ts`.
- `~/.venvs/lensjudge/.../matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf` → `render/fonts/`.

## Risks → mitigations
- **SDK bundled-CLI drift** (0.2.96 spawns 2.1.172 unless `cli_path`): pin `claude-agent-sdk>=0.2.148`, default `cli_path=which("claude")`, `doctor` prints the binary; full model ids only.
- **Global `~/.claude/settings.json` (`effortLevel: xhigh`, `model: fable[1m]`) leaking**: `setting_sources=[]` always; fake-CLI test asserts `--setting-sources=`.
- **Effort on Haiku errors**: selector greys effort for `claude-haiku-4-5`.
- **Pillow/FreeType change breaks goldens**: pinned `pillow==12.2.0`; `--update-golden` + diff diagnostics.
- **SVG preview ≠ PIL render**: shared `style_defaults`; preview labelled "preview"; screenshot-vs-render tolerance check in `m2_render.sh` catches flips/dash phase/missing legend.
- **Unknown pixel scale of the deck tiles**: `cutout_arcsec` per campaign config with per-image override; UI shows derived scale with an `assumed` badge; deck tiles are fixtures, not few-shot exemplars.
- **Fable long turns / structured-output retries exhausted**: SSE keepalive + cancel; `max_budget_usd`; parse fallback + one repair; `parse_ok` counted.
- **`.playwright-cli/` artefacts** (60 MB at repo root today): run QA from `apps/LensMark`; gitignore there.

## Findings to relay (not in scope, worth fixing in lensjudge separately)
- `imaging/grader_lean.py:146-147` passes `setting_sources=None` (= load ALL settings), so the global `effortLevel: xhigh` has been leaking into lensjudge runs; `[]` is isolation.
- The installed Python SDK 0.2.96 runs lensjudge through its bundled CLI 2.1.172, not the PATH 2.1.251; the alias `opus` may not resolve to `claude-opus-5` there.
