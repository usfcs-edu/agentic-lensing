# LensMark geometry + DOM contract (shared by `lensmark/render/` and `frontend/`)

Both the PIL renderer (canonical `.annot.png`) and the SVG preview draw from the same
`LensMarkFile` with the same rules. Constants come from `file.style_defaults` (== `lensmark/schema/style_defaults.json`,
also served at `GET /api/style`) and `palette.json`. `m = min(W, H)`; every style value in
`fraction_of_min_dim` units is multiplied by `m` to get pixels. Positions are `[u, v]` → `(u·W, v·H)`.

## Items
| type | geometry | rule |
|---|---|---|
| `arrow` | shaft from `tail` towards `head`, stopping `arrow.tip_gap·m` short of `head`; filled triangular head of length `head_len·m`, base width `head_w·m`, apex at the gap point; stroke `line_w·m`, colour `palette[color]` | **label**: `label_anchor` `tail` (default for `auto`): text centred on a point `label.offset·m` beyond the tail along the head→tail direction; `head`: same beyond the head; `auto` falls back to `head` side only when the tail-side box would leave the image. Font `label.size·m`, bold, colour = arrow colour, dark halo `label.halo` of `halo_px`. Label box is clamped inside the image (margin 4 px). `label_offset` (u,v fractions) is added after placement. |
| `mask_circle` | circle at `center`, radius `radius_arcsec / pixel_scale_arcsec` px | `kind=galaxy` → dashed (`mask_galaxy.dash_len·m` on, `gap_len·m` off); `kind=star` → dotted (dots of radius `mask_star.dot_r·m`, edge gap `gap_mult·dot_r·m`, i.e. centre spacing `(2+gap_mult)·dot_r·m`); `kind=artifact` → dashed with half dash length. Dash/dot phase starts at screen angle 0 (rightmost point) and walks clockwise on screen. Colour `mask_red`, stroke `line_w·m`. No label. |
| `einstein_ring` | circle at `center` (or at the head of `center_ref` arrow when set), radius `theta_e_arcsec / pixel_scale_arcsec` | fine dots (`einstein_ring.dot_r·m`, spacing `(2+gap_mult)·dot_r·m`), colour `ring_white`. θ label: `label` if set else `θ_E ≈ {theta_e_arcsec:.2g}″`; at `label_pos` if set else centred `theta_label.offset·m` below-right of the ring (angle −45° screen, distance `r + offset·m`); font `theta_label.size·m`, italic-ish allowed, colour white with halo. |
| `text` | `text` at `pos` | font `text.size·m`, colour `palette[color]`, halo. |
| legend | `legend.show` true → plate with one row `→ label` per item with `show_in_legend` and a `label`, coloured like the item, in `legend.order` or items order | plate bg `legend.bg`, pad `legend.pad·m`, font `legend.size·m`, line height `line_h × font size`. `position` `auto` → the corner (of the four) whose quadrant contains the fewest item anchor points (arrow tail/head, circle centres); ties → `top_left`. Plate inset 2·pad from the edges. |

Items with `status` `rejected` or `invalid` are **not** drawn in the canonical render. `proposed` items ARE drawn by
the canonical render (so a proposal can be inspected as a PNG) but the browser shows them in the ghost layer.
Draw order: mask circles → einstein ring → arrows → text → labels → legend. In the PIL renderer, labels are drawn
on a layer *below* geometry (labels never hide a mark); the SVG preview may simply draw labels last.

## Deterministic PIL render
3× supersampled RGBA overlay, `Image.LANCZOS` downsample, alpha-composite on the original, PNG `optimize=False`,
no metadata; output size = `image.width × image.height` (× `scale` for export only). Fonts only from
`lensmark/render/fonts/` (never system fonts). On render, the JSON gets `render = {renderer, output, of_json_sha256, rendered_at}`
where `of_json_sha256 = file.content_sha256()` and the JSON is saved with `touch_modified=False`.

## Browser DOM (playwright-cli asserts on this)
```
<div id="stage"><img id="base" src=/api/images/{id}/original> <svg id="overlay" viewBox="0 0 W H"> ... </svg></div>
  <g id="items">   committed items (status accepted|edited; rejected/invalid hidden unless "show rejected")
  <g id="ghost">   proposed items (status proposed), drawn dashed-outline at 50% opacity
  <g id="legend">  legend plate
each item: <g class="annot {arrow|mask galaxy|mask star|mask artifact|ring|text}" data-id data-type data-status [data-kind]>
  arrow: <line class="shaft"/> <polygon class="head"/> <text class="label"/>
  mask : <circle class="mask-circle" stroke-dasharray="..."/>          (dash: "dash gap"; dot: "0 spacing" with stroke-linecap round and stroke-width 2·dot_r·m)
  ring : <circle class="ring-circle"/> <text class="theta-label"/>
  text : <text class="note"/>
selection handles: <g id="handles"> with <circle class="handle" data-handle="tail|head|center|radius|pos">
```
`data-testid` names the UI must expose: `image-row` (with `data-id`, `data-status`), `tool-select`, `tool-arrow`, `tool-galaxy`,
`tool-star`, `tool-ring`, `tool-text`, `save`, `dirty`, `item-row` (`data-id`, `data-status`), `label-input`, `color-select`,
`legend-checkbox`, `delete-item`, `description`, `theta-e-input`, `grade-select`, `system-verdict`, `render-toggle`,
`model-select`, `effort-select`, `budget-input`, `propose`, `propose-cancel`, `propose-log`, `accept-all`, `reject-all`,
`accept-item`, `reject-item`, `verdict-select`, `review-comment`, `submit-critique`, `critique-panel`, `voice-text`, `voice-send`,
`voice-mic`, `apply-all`, `apply-op`, `reject-op`, `export-coco`, `export-ds9`, `export-masks`, `export-fewshot`, `lint`.
Test hook: `window.__lensmark = { state, addItem(json), selectItem(id), save(), load(id), setTool(name) }`.

## Keyboard
`A` arrow (drag tail→head), `G` galaxy mask (drag centre→edge), `S` star mask, `R` ring (click the centre; radius from the
θ_E input), `T` text (click), `V`/`Esc` select, `Delete`/`Backspace` delete selection, `Cmd/Ctrl+Z` undo, `Cmd/Ctrl+S` save,
`[`/`]` previous/next image, `Z` zoom 2× toggle, `L` cycle legend corner, `1`–`7` critique verdicts on the selected
proposed item (correct, wrong_position, wrong_label, wrong_type, wrong_size, spurious, redundant), `Enter` accept, `X` reject.
