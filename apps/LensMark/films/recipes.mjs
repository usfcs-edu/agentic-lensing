/**
 * LensMark film — what each scene SHOWS (script.mjs says what each scene CLAIMS).
 * Shots per the pipeline's four kinds; acts drive the real app at http://127.0.0.1:8765
 * (start it first: LENSMARK_ENGINE=sdk ~/.venvs/lensmark/bin/python -m lensmark.cli serve
 *  films/campaign-film --port 8765 --no-open).
 * All hold() fractions below are MEASURED (node measure.mjs segments lensmark <scene>).
 */
import { defineShots, defineAction } from './shots.mjs'

const APP = 'http://127.0.0.1:8765'

defineShots('lensmark', {
  capture: 'lensmark',
  header: 'apps/LensMark — a real run',
  scenes: {
    'a1-card': {
      kind: 'card',
      title: 'LensMark',
      subtitle: 'hand · model · voice — one accountable file format',
      note: 'filmed live against a running instance',
    },
    'a2-tour':    { kind: 'web', url: APP + '/#deck-02', ready: 'LensMark' },
    'a3-arrow':   { kind: 'web', url: APP + '/#deck-01', ready: 'LensMark' },
    'a4-ring':    { kind: 'web', url: APP + '/#deck-01', ready: 'LensMark' },
    'a5-render':  { kind: 'web', url: APP + '/#deck-01', ready: 'LensMark' },
    'a6-files': {
      kind: 'terminal',
      prompt: 'ls deck-02* deck-03*;  lensmark render films/campaign-film --check',
      from: 'FILES', to: 'EVAL', max: 22, highlight: ['STALE'],
    },
    'a7-propose': { kind: 'web', url: APP + '/#deck-03', ready: 'LensMark' },
    'a8-review':  { kind: 'web', url: APP + '/#deck-03', ready: 'LensMark' },
    'a9-eval': {
      kind: 'terminal',
      prompt: 'lensmark eval films/campaign-film --by model,effort  &&  python films/eval_view.py  (headline columns)',
      from: 'EVAL', max: 10, highlight: ['claude-opus-5'],
      // static: the narration names these columns while they are spoken; a paced reveal lands
      // the last row ~2s late (video-start precedes the reveal timer by the spawn cost)
      reveal: false,
    },
    'a10-voice':  { kind: 'web', url: APP + '/#deck-01', ready: 'LensMark' },
    'a11-export': { kind: 'web', url: APP + '/#deck-01', ready: 'LensMark' },
    'a12-close': {
      kind: 'card',
      title: 'LensMark',
      subtitle: 'hand · model · voice — one accountable format',
      note: 'every call live · every frame a real screen',
    },
  },
})

// ─── shared act prelude ───────────────────────────────────────────────────────────────
// Injected into every web act. Waits for the app, dismisses a stray draft bar, adds a
// visible fake cursor (headless clicks draw nothing), and defines the helpers.
const PRELUDE = `
  await page.locator('[data-testid=image-row]').first().waitFor({ timeout: 20000 })
  const __dd = page.locator('[data-testid=discard-draft]')
  if (await __dd.count()) { try { await __dd.click({ timeout: 800 }) } catch {} }
  await page.evaluate(() => {
    if (document.getElementById('__cur')) return
    const d = document.createElement('div'); d.id = '__cur'
    d.style.cssText = 'position:fixed;z-index:2147483647;width:18px;height:18px;border-radius:50%;' +
      'background:rgba(255,214,74,.95);border:2px solid rgba(0,0,0,.7);box-shadow:0 1px 6px rgba(0,0,0,.6);' +
      'pointer-events:none;left:960px;top:600px;transition:left .3s ease,top .3s ease;margin:-9px 0 0 -9px'
    document.body.appendChild(d)
  })
  const cur = async (x, y) => {
    await page.evaluate(([a, b]) => { const d = document.getElementById('__cur'); d.style.left = a + 'px'; d.style.top = b + 'px' }, [x, y])
    await page.mouse.move(x, y, { steps: 12 })
    await page.waitForTimeout(320)
  }
  const uv = async (u, v) => {
    const r = await page.evaluate(() => { const b = document.querySelector('#overlay').getBoundingClientRect()
      return { x: b.x, y: b.y, w: b.width, h: b.height } })
    return [r.x + u * r.w, r.y + v * r.h]
  }
  const dragUV = async (u0, v0, u1, v1) => {
    const [x0, y0] = await uv(u0, v0); const [x1, y1] = await uv(u1, v1)
    await cur(x0, y0); await page.mouse.down(); await cur(x1, y1); await page.waitForTimeout(150); await page.mouse.up()
  }
  const click = async (sel) => {
    try { await page.locator(sel).first().scrollIntoViewIfNeeded({ timeout: 2000 }) } catch {}
    const box = await page.locator(sel).first().boundingBox()
    if (!box) { await page.evaluate(s => { document.title = 'ACT-FAILED missing ' + s }, sel); throw new Error('missing ' + sel) }
    await cur(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.down(); await page.mouse.up()
  }
  const hover = async (sel) => {
    try { await page.locator(sel).first().scrollIntoViewIfNeeded({ timeout: 2000 }) } catch {}
    const box = await page.locator(sel).first().boundingBox()
    if (box) await cur(box.x + box.width / 2, box.y + box.height / 2)
  }
  const blur = async () => { await page.evaluate(() => document.activeElement && document.activeElement.blur()) }
  const tab = async (name) => { await click('[data-testid=tab-' + name + ']') }
  const saved = async () => { await page.waitForFunction(() => !window.__lensmark.state.dirty, null, { timeout: 12000 }) }
  const fail = async (why) => { await page.evaluate(w => { document.title = 'ACT-FAILED ' + w }, why); throw new Error(why) }
  const done = async (ok, why) => { await page.evaluate(t => { document.title = t }, ok ? 'OK' : 'ACT-FAILED ' + (why || '')) }
`

// a2-tour (17.11s) — measured beats:
// 0.000 "A campaign is a folder of cutouts"   idle on stage
// 0.169 "nine of them here, listed"           walk the image rows
// 0.406 "the stage in the centre"             cursor to stage
// 0.495 "working panels on the right"         trace the tab strip
// 0.641 "Two images are blank on purpose"     rest on deck-01 row
// 0.793 "one…by hand…one…for the model"       rest on deck-03 row
defineAction('lensmark', 'a2-tour', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('items')
  await cur(...(await uv(0.5, 0.45)))
  await hold(0.169)
  for (let i = 0; i < 4; i++) {
    const b = await page.locator('[data-testid=image-row]').nth(i).boundingBox()
    if (b) await cur(b.x + b.width / 2, b.y + b.height / 2)
  }
  await hold(0.406); await cur(...(await uv(0.5, 0.5)))
  await hold(0.495); await hover('[data-testid=tab-export]')
  await hold(0.641); await hover('[data-testid=image-row][data-id=deck-01]')
  await hold(0.793); await hover('[data-testid=image-row][data-id=deck-03]')
  const seen = await inFrame('[data-testid=image-row][data-id=deck-03]', 'the blank deck-03 row')
  await hold(0.86); await done(seen.ok, seen.why)
}`)

// a3-arrow (16.92s):
// 0.149 "A is the arrow"  press a · 0.225 "drag from tail to head" · 0.325 "then label it"
// 0.383 "this tight blue arc…"  colour cyan lands mid-phrase · 0.615 "G drags out a galaxy mask…"
// ~0.80 silent Save (state must carry to a4)
defineAction('lensmark', 'a3-arrow', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('items')
  await hold(0.149); await blur(); await page.keyboard.press('a')
  await hold(0.225); await dragUV(0.451, 0.646, 0.451, 0.567)
  await hold(0.325); await click('[data-testid=label-input]'); await page.keyboard.type('tight arc', { delay: 55 })
  await hold(0.47); await hover('[data-testid=color-select]')
  await page.locator('[data-testid=color-select]').first().selectOption('cyan', { timeout: 4000 })
  await hold(0.615); await blur(); await page.keyboard.press('g')
  await dragUV(0.033, 0.362, 0.115, 0.362)
  await page.keyboard.press('Escape')
  if (await page.locator('#items .annot.arrow').count() < 1) await fail('arrow not drawn')
  if (await page.locator('#items .annot.mask.galaxy').count() < 1) await fail('galaxy mask not drawn')
  await hold(0.80); await blur(); await click('[data-testid=save]'); await saved()
  await hold(0.86); await done(true)
}`)

// a4-ring (16.53s):
// 0.05 "S drops a dotted star mask" · 0.161 "R places the Einstein ring…"
// 0.421 "one and a half arcseconds here" — theta label must be in frame
// 0.639 "Save commits it all" · tail: three-artefact sentence over the saved state
defineAction('lensmark', 'a4-ring', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('items')
  await hold(0.05); await blur(); await page.keyboard.press('s')
  await dragUV(0.393, 0.245, 0.416, 0.245)
  await hold(0.161); await blur(); await page.keyboard.press('r')
  const [rx, ry] = await uv(0.506, 0.496)
  await cur(rx, ry); await page.mouse.down(); await page.mouse.up()
  await hold(0.421)
  const lbl = await inFrame('#items .annot.ring .theta-label', 'the theta_E label')
  if (!lbl.ok) await fail(lbl.why)
  await hold(0.639); await blur(); await click('[data-testid=save]'); await saved()
  await hold(0.86); await done(true)
}`)

// a5-render (15.85s): 0.327 "Toggle Rendered" · rendered PNG holds the frame · 0.85 toggle back
defineAction('lensmark', 'a5-render', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('items')
  await cur(...(await uv(0.5, 0.4)))
  await hold(0.327); await click('[data-testid=render-toggle]')
  await page.waitForFunction(() => { const b = document.querySelector('#base'); return b && b.src.includes('/annot') && b.naturalWidth === 403 }, null, { timeout: 12000 })
  const seen = await inFrame('#base', 'the rendered PNG')
  await hold(0.85); await click('[data-testid=render-toggle]')
  await hold(0.86); await done(seen.ok, seen.why)
}`)

// a7-propose (13.82s): 0.117 "Pick a model" sonnet · 0.21 "an effort level" low
// 0.267 "Propose makes a live call" — click; the REAL stream fills the log to the cut
defineAction('lensmark', 'a7-propose', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('propose')
  await hold(0.117); await hover('[data-testid=model-select]')
  await page.locator('[data-testid=model-select]').first().selectOption('sonnet', { timeout: 4000 })
  await hold(0.21); await hover('[data-testid=effort-select]')
  await page.locator('[data-testid=effort-select]').first().selectOption('low', { timeout: 4000 })
  await hold(0.267); await click('[data-testid=propose]')
  await page.waitForFunction(() => ((document.querySelector('[data-testid=propose-log]') || {}).textContent || '').length > 3, null, { timeout: 10000 })
  const seen = await inFrame('[data-testid=propose-log]', 'the streaming log')
  await hold(0.86); await done(seen.ok, seen.why)
}`)

// a8-review (20.86s): poll the server-side merge (absorbs API latency), then
// 0.229 Review tab · 0.320 accept an item · 0.496 "spurious, in this case" verdict+reject
// 0.712 "Submitting the critique" · result line to the cut
defineAction('lensmark', 'a8-review', `async (page, hold, inFrame) => {
  ${PRELUDE}
  let ghosts = 0
  for (let i = 0; i < 30 && !ghosts; i++) {
    await page.evaluate(() => window.__lensmark.load('deck-03'))
    await page.waitForTimeout(600)
    ghosts = await page.locator('#ghost .annot').count()
  }
  if (!ghosts) await fail('no ghosts — re-propose deck-03 (sonnet/low) and re-record a7,a8')
  await hold(0.229); await tab('review')
  await hold(0.320); await click('[data-testid=review-row][data-status=proposed] [data-testid=accept-item]')
  await hold(0.496)
  await hover('[data-testid=review-row][data-status=proposed] [data-testid=verdict-select]')
  await page.locator('[data-testid=review-row][data-status=proposed] [data-testid=verdict-select]').first().selectOption('spurious', { timeout: 4000 })
  await hold(0.60); await click('[data-testid=review-row][data-status=proposed] [data-testid=reject-item]')
  await hold(0.712); await click('[data-testid=submit-critique]')
  await page.waitForFunction(() => ((document.querySelector('[data-testid=critique-result]') || {}).textContent || '').length > 0, null, { timeout: 15000 })
  await hold(0.86); await done(true)
}`)

// a10-voice (21.14s): 0.106 hover the mic (shown, not clicked — disclosed in narration)
// 0.236 type the transcript (during "a film set is a poor room for dictation")
// ~0.35 Send — REAL patch call, 5–12 s measured; poll ≤ 13 s; the op row must still be on screen
// at 0.768 ('a rationale and a confidence'), so Apply fires at 0.85, after the phrase
// 0.768 "a rationale and a confidence" — op rows on screen
defineAction('lensmark', 'a10-voice', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('voice')
  const n0 = await page.locator('#items .annot.mask.star').count()
  await hold(0.106); await hover('[data-testid=voice-mic]')
  await hold(0.236); await click('[data-testid=voice-text]')
  await page.keyboard.type('put a dotted circle around the star at the upper right', { delay: 26 })
  await hold(0.35); await click('[data-testid=voice-send]')
  let ops = 0
  for (let i = 0; i < 26 && !ops; i++) { await page.waitForTimeout(500); ops = await page.locator('[data-testid=apply-op]').count() }
  if (!ops) await fail('patch ops never arrived — re-record a10')
  const seen = await inFrame('[data-testid=voice-op]', 'a voice op row')
  await hold(0.85); await click('[data-testid=apply-all]')
  await page.waitForFunction(n => document.querySelectorAll('#items .annot.mask.star').length > n, n0, { timeout: 12000 })
  await hold(0.86); await done(seen.ok, seen.why)
}`)

// a11-export (13.38s): 0.245 COCO · 0.410 DS9 · 0.635 masks · 0.742 few-shot; file list fills
defineAction('lensmark', 'a11-export', `async (page, hold, inFrame) => {
  ${PRELUDE}
  await tab('export')
  await hold(0.245); await click('[data-testid=export-coco]')
  await page.waitForFunction(() => document.querySelectorAll('[data-testid=export-files] li').length > 0, null, { timeout: 20000 })
  await hold(0.410); await click('[data-testid=export-ds9]')
  await hold(0.635); await click('[data-testid=export-masks]')
  await hold(0.742); await click('[data-testid=export-fewshot]')
  const seen = await inFrame('[data-testid=export-files]', 'the written files list')
  await hold(0.86); await done(seen.ok, seen.why)
}`)
