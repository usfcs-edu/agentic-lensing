# LensMark HTTP contract (`lensmark serve DIR` → `http://127.0.0.1:8765`)

The browser front end talks to the core **only** through these routes; the CLI mirrors them 1:1.
All JSON responses are `Cache-Control: no-store`. Errors are `{"error": "...", "detail": ...}` with 4xx/5xx.
`{id}` is the image stem (`[A-Za-z0-9_.-]+`).

| Method | Path | Body → Response |
|---|---|---|
| GET | `/` | the app (`lensmark/static/index.html`) |
| GET | `/api/health` | `{version, campaign_dir, engine, claude_bin, claude_version, n_images}` |
| GET | `/api/models` | `{models:[{alias,id,label,supports_effort,price_in,price_out}], efforts:[...], default:{model,effort}}` |
| GET | `/api/style` | `{palette:{colors,arrow_order,deflector,reserved}, style_defaults:{...}}` — the same constants the PIL renderer uses |
| GET | `/api/schema/{lensmark\|proposal\|critique\|patch}` | JSON Schema |
| GET | `/api/config` | campaign config (`lensmark.config.json` merged with defaults) |
| GET | `/api/images` | `[summary…]` — `Campaign.summary(id)`: `{id,file,width,height,cutout_arcsec,scale_source,has_json,has_annot,annot_stale,n_items,by_status,grade,verdict,theta_e_arcsec,rank,modified,n_proposals}` |
| GET | `/api/images/{id}/original` | image bytes (`ETag` = sha256) |
| GET | `/api/images/{id}/annot` | rendered PNG (renders on demand if stale/missing) |
| GET | `/api/images/{id}/thumb?px=160` | small JPEG thumbnail |
| GET | `/api/ann/{id}` | the `LensMarkFile` JSON; if none is saved yet, a fresh unsaved one (`X-LensMark-Exists: 0`) |
| PUT | `/api/ann/{id}` | body = full `LensMarkFile` JSON → validates (`extra=forbid`), atomic write, log diff, re-render → `{ok, modified, render:{of_json_sha256,output}, lint:[...]}` |
| GET | `/api/ann/{id}/log` | `[log events…]` |
| POST | `/api/render/{id}` | `{}` → `{output, sha256, stale:false}` |
| POST | `/api/propose/{id}` | `{model?, effort?, budget?, fewshot?}` → **202** `{run_id}` (starts a background task) |
| GET | `/api/propose/{id}/{run_id}/events` | **SSE**: `data: {"phase": "queued\|started\|thinking\|partial\|tool\|validated\|done\|error", "detail": str, "text"?: str, "cost_usd"?: n, "n_items"?: n, "run": ProposalRun?}`; keepalive comment every 15 s; ends after `done`/`error` |
| POST | `/api/propose/{id}/{run_id}/cancel` | → `{ok}` |
| GET | `/api/proposals/{id}` | `[ProposalRun…]` (from `provenance.proposal_runs` + files in `proposals/`) |
| GET | `/api/proposals/{id}/{run_id}` | the immutable proposal file |
| POST | `/api/critique/{id}` | body = `Critique` JSON (+ the reviewed `LensMarkFile` is PUT separately by the UI) → `{file}`; merges verdicts into the item `review` fields, appends `provenance.critiques` |
| GET | `/api/eval?by=model,effort` | `{rows:[…]}` (same as `lensmark eval`) |
| POST | `/api/export/{coco\|ds9\|masks\|fewshot}` | `{ids?, k?, require_flag?}` → `{files:[…]}` under `exports/` |
| POST | `/api/patch/{id}` | `{transcript, model?, effort?}` → `Patch` JSON (ops are NOT applied) |
| POST | `/api/patch/{id}/apply` | `{ops:[…], transcript?}` → updated `LensMarkFile` (applied + saved + logged `source:"voice"`) |
| POST | `/api/stt` | multipart `audio` → `{transcript, backend}` (501 if no STT backend configured) |

Proposal flow in the UI: `POST /api/propose/{id}` → open the SSE → on `done`, `GET /api/ann/{id}`
again (the proposed items are merged into the file with `status:"proposed"` and a `ProposalRun`
appended) and draw them in the ghost layer. Critique flow: the UI edits item `status`/`review` locally,
PUTs the file, then POSTs the `Critique` document.
