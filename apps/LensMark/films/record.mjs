/**
 * Drive Chromium and film each scene for exactly as long as its narration takes.
 *
 * The inversion this pipeline is built on: `narrate.mjs` has already synthesized the words
 * and measured them into `out/timing.json`. Nothing here decides how long a shot lasts —
 * the sentence does. Nothing is trimmed to fit afterwards.
 *
 * ─── FIVE THINGS ABOUT playwright-cli, EACH OF WHICH COST A TAKE ──────────────────────
 *
 * 1. HOW IT REPORTS A THROWN PROGRAM HAS CHANGED ONCE ALREADY. It has always announced the
 *    failure in the TEXT, as a `### Error` block; it used to exit 0 alongside that, and
 *    since 0.1.18 it exits 1. This file once read the text ALONE, on the stated grounds
 *    that the exit code was worthless — so when the exit code started being set, the
 *    wrapper's `execFileSync` threw and discarded the very text every guard depended on.
 *    BOTH signals are now checked, neither is load-bearing by itself, and
 *    `assertErrorChannelWorks()` proves the text channel still exists before a single frame
 *    is recorded. The package is pre-1.0 and installed unpinned; assume it will move again.
 * 2. `run-code` TAKES A BARE FUNCTION EXPRESSION — `async (page) => { … }`. A program body
 *    fails with `SyntaxError: Unexpected identifier`.
 * 3. `console.log` INSIDE `run-code` IS NOT SURFACED. Everything comes back through
 *    `document.title`.
 * 4. A HEADLESS *SHELL* CANNOT RECORD. It has no compositor: it writes nothing, reports no
 *    error, and leaves a plausible-looking empty file. Asserted below rather than trusted.
 * 5. `file:` IS BLOCKED. The generated pages are served over loopback instead — by a
 *    SEPARATE PROCESS, because `execFileSync` blocks this one's event loop and an
 *    in-process server would deadlock against the navigation it exists to serve.
 *
 *   node record.mjs                       # every film
 *   node record.mjs <filmId>              # one film
 *   node record.mjs <filmId> a3,a4,a5     # repair a bad tail without re-shooting
 */
import { execFileSync, spawn, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { FILMS } from './script.mjs'
import { ACTS, prepareFilm } from './shots.mjs'
import { LIMITS, RUNTIME, VIDEO } from './film.config.mjs'
// Side-effect import: recipes.mjs registers every film's shots with shots.mjs. Importing it
// here rather than from shots.mjs keeps the mechanism free of a cycle back to itself.
import './recipes.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out')
const ASSETS = join(HERE, 'assets')
const CONFIG = join(HERE, '.playwright', 'cli.config.json')
const VIDEO_SIZE = `${VIDEO.width}x${VIDEO.height}`

mkdirSync(OUT, { recursive: true })

// ── locate the CLI ───────────────────────────────────────────────────────────────────────
// Local install first: setup.sh puts it there so no admin is needed, and a local install
// pins the browser revision to the CLI that will drive it.
function resolveCli() {
  if (RUNTIME.playwrightCli) return RUNTIME.playwrightCli
  const local = join(HERE, 'node_modules', '.bin', 'playwright-cli')
  if (existsSync(local)) return local
  return 'playwright-cli'
}
const PWCLI = resolveCli()

/**
 * A genuinely blocking sleep, with no child process and no platform assumption.
 *
 * The pipelines this generalizes shelled out to `/bin/sleep`, which costs a process per
 * quarter-second and does not exist on Windows. `Atomics.wait` on a SharedArrayBuffer
 * nobody else holds blocks this thread and only this thread.
 */
function sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)
}

// ── the guard that pays for itself ───────────────────────────────────────────────────────
if (!existsSync(CONFIG)) {
  throw new Error(`No browser config at ${CONFIG}. Run the skill's scripts/setup.sh.`)
}
const config = JSON.parse(readFileSync(CONFIG, 'utf8'))
const execPath = config?.browser?.launchOptions?.executablePath ?? ''
if (/headless_shell/.test(execPath)) {
  throw new Error(
    `REFUSING to record: ${CONFIG} points at a headless shell (${execPath}).\n` +
    'It has no compositor — it records nothing, reports no error, and writes a ' +
    'plausible-looking empty file. Point executablePath at a full Chromium.')
}
if (!existsSync(execPath)) throw new Error(`Chromium not found at ${execPath} (see ${CONFIG})`)

// ── the CLI wrapper ──────────────────────────────────────────────────────────────────────
function cli(args, { tolerate = false } = {}) {
  // `spawnSync`, not `execFileSync`, and the distinction is the whole bug: `execFileSync`
  // THROWS AWAY everything the process said when it exits non-zero. Since playwright-cli
  // 0.1.18 a thrown program is a non-zero exit, so the `### Error` block went into the
  // exception and out of reach — leaving `tolerate` nothing to inspect and making a
  // deliberately-throwing probe indistinguishable from a missing binary.
  //
  // A generous buffer: a snapshot block is large, and a truncated read would hide the very
  // marker this function exists to find.
  const run = spawnSync(PWCLI, [`-s=${RUNTIME.session}`, ...args], {
    encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
  })
  // `run.error` is the process never starting — a missing binary, not a failing run. That
  // is never tolerable, whatever the caller asked for.
  if (run.error) throw run.error
  const said = `${run.stdout ?? ''}${run.stderr ?? ''}`
  if (!tolerate && (/^### Error/m.test(said) || run.status !== 0)) {
    const at = said.indexOf('### Error')
    const detail = (at >= 0 ? said.slice(at) : said).split('\n').slice(0, 12).join('\n')
    throw new Error(`playwright-cli ${args[0]} reported an error (exit ${run.status}):\n${detail}`)
  }
  return said
}

/** Run a bare function expression in the page; returns whatever it put in `document.title`. */
function inPage(source) {
  const file = join(OUT, '_program.js')
  writeFileSync(file, source, 'utf8')
  const said = cli(['run-code', '--filename', file])
  return /- Page Title: (.*)/.exec(said)?.[1]?.trim() ?? ''
}

/** `eval` returns a text blob; pull the value back out of it. */
function evaluate(js) {
  const said = cli(['eval', js])
  const m = /### Result\s*\n([\s\S]*)$/.exec(said)
  return (m ? m[1] : said).trim()
}

/**
 * Prove the error channel before trusting it.
 *
 * If playwright-cli ever stops printing `### Error`, every check in this file silently
 * becomes a no-op and the films record whatever happened to be on screen. This makes that a
 * refusal instead of a surprise.
 */
function assertErrorChannelWorks() {
  const file = join(OUT, '_boom.js')
  writeFileSync(file, 'async (page) => { throw new Error("deliberate") }', 'utf8')
  const said = cli(['run-code', '--filename', file], { tolerate: true })
  if (!/### Error/m.test(said)) {
    throw new Error(
      'REFUSING to record: playwright-cli no longer prints "### Error" when a program throws, ' +
        'so every guard in this file has silently stopped working. Fix the detection first.')
  }
}

/**
 * Is the thing the narration names actually IN THE PICTURE?
 *
 * Handed to every act as its third argument. Every other check reads the whole document and
 * therefore cannot tell the difference between "on screen" and "in the DOM two thousand
 * pixels below the fold" — which is exactly how a closing shot once narrated a row that was
 * at y=2221 in a 720-pixel frame, passing an `innerText().includes(name)` check on every
 * take.
 */
const IN_FRAME_HELPER = `
  const inFrame = async (target, label) => {
    const what = label || (typeof target === 'string' ? target : 'the element')
    const loc = typeof target === 'string' ? page.locator(target) : target
    const n = await loc.count().catch(() => 0)
    if (n === 0) return { ok: false, why: what + ': nothing in the DOM matches it at all' }
    const el = loc.first()

    // THE RECTANGLE COMES FROM boundingBox(), NOT getBoundingClientRect(), AND THAT IS THE
    // WHOLE DIFFERENCE FOR AN ELEMENT INSIDE AN IFRAME. \`getBoundingClientRect()\` is
    // relative to the document it is called in, so an anchor inside an iframe would be
    // measured against the IFRAME's box and its \`window.innerWidth\` — reporting a link as
    // "in frame" while the iframe itself sat off the bottom of the picture. Playwright's
    // \`boundingBox()\` returns MAIN-FRAME coordinates with every ancestor frame's offset
    // already applied, so one code path is right for both. It returns null for an element
    // with no rendered box at all, which is \`display:none\` and the ones like it.
    const box = await el.boundingBox().catch(() => null)
    const vp = page.viewportSize() || { width: 0, height: 0 }
    if (!box) return { ok: false, why: what + ': in the DOM but renders no box at all' }

    // Style and occlusion are properties of the element's OWN document, so they are
    // measured there. \`elementFromPoint\` inside an iframe therefore sees only what could
    // cover it inside that iframe; an overlay on the parent page is not detected. Documented
    // rather than papered over — that second case has never been the failure mode here.
    const said = await el.evaluate(node => {
      const r = node.getBoundingClientRect()
      const s = window.getComputedStyle(node)
      const cx = r.left + r.width / 2
      const cy = r.top + r.height / 2
      const onScreen = cx >= 0 && cy >= 0 && cx < window.innerWidth && cy < window.innerHeight
      const hit = onScreen ? document.elementFromPoint(cx, cy) : null
      return JSON.stringify({
        styled: s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) === 0,
        covered: !hit || !(hit === node || node.contains(hit) || hit.contains(node)),
      })
    }).catch(() => '')
    const flags = said ? JSON.parse(said) : { styled: false, covered: false }

    const x = Math.round(box.x), y = Math.round(box.y)
    const w = Math.round(box.width), h = Math.round(box.height)
    const where = ' [' + x + ',' + y + ' ' + w + 'x' + h + ' in ' + vp.width + 'x' + vp.height + ']'
    if (w <= 0 || h <= 0) return { ok: false, why: what + ': zero-sized' + where }
    if (flags.styled) return { ok: false, why: what + ': present but styled invisible' + where }
    if (x < 0 || y < 0 || x + w > vp.width || y + h > vp.height) {
      return { ok: false, why: what + ': OUTSIDE THE CAPTURED FRAME' + where }
    }
    if (flags.covered) return { ok: false, why: what + ': in the frame but covered' + where }
    return { ok: true, why: what + ': in frame' + where }
  }
`

// ── a loopback static server, IN ITS OWN PROCESS ─────────────────────────────────────────
function serveDir(root, port) {
  const child = spawn(process.execPath, [join(HERE, 'serve.mjs'), root, String(port), ASSETS],
                      { stdio: ['ignore', 'pipe', 'inherit'] })
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`static server did not start on ${port}`)), 10_000)
    child.stdout.on('data', d => {
      if (String(d).includes('listening')) { clearTimeout(timer); resolve(child) }
    })
    child.once('error', reject)
    child.once('exit', code => reject(new Error(`static server exited with ${code}`)))
  })
}

/**
 * Record the current screen for `seconds`.
 *
 * The wait happens IN THE PAGE rather than in this process: the recorder captures the
 * browser, so the browser has to be the thing that is busy for that long. A `setTimeout`
 * here would stop the video at the right moment having filmed a page nobody touched — the
 * same footage for the wrong reason, and it would not survive the browser being slow to
 * start the capture.
 */
function record(path, seconds, { act = null, reveal = false } = {}) {
  cli(['video-start', path, '--size', VIDEO_SIZE])

  let spent = 0
  if (act) {
    const began = Date.now()
    // `hold(fraction)` is how an act STAYS somewhere until a given point in its own
    // narration. An act that runs flat out finishes in two seconds and leaves the narration
    // describing a screen that is no longer there. The fraction is of THIS scene's
    // narration, so an act reads as a rough storyboard: `await hold(0.4)` means "keep this
    // on screen until we are 40% through what is being said over it".
    const paced = `async (page) => {
      const __began = Date.now()
      const __ms = ${Math.round(seconds * 1000)}
      const hold = async fraction => {
        const left = (__began + Math.round(fraction * __ms)) - Date.now()
        if (left > 0) await page.waitForTimeout(left)
      }
      ${IN_FRAME_HELPER}
      return await (${act})(page, hold, inFrame)
    }`
    const said = inPage(paced)
    spent = (Date.now() - began) / 1000
    if (said.startsWith('ACT-FAILED')) {
      cli(['video-stop'], { tolerate: true })
      throw new Error(`${path}: the on-camera action failed — ${said}`)
    }
  } else if (reveal) {
    // Fire-and-forget: __reveal schedules the lines and returns immediately, so the wait
    // below is what actually keeps the browser busy while they arrive.
    evaluate(`() => window.__reveal ? window.__reveal(${seconds}) : 0`)
  }

  const remaining = seconds - spent

  // TWO failures, not one, and the quiet one is the one that bites.
  //
  // OVERRUN is loud: the act outlasts its narration and the tail gets cut.
  //
  // UNDERRUN is silent and far more common. An act that finishes in the first seconds leaves
  // the rest of the scene frozen on whatever it ended on while the narration is still
  // describing what it did — an export dialog quoted verbatim for fourteen seconds after it
  // was dismissed. The footage is not wrong; it is the AFTERMATH of the thing being
  // described, which reads as a slideshow pretending to be a demonstration.
  //
  // Not fatal: a scene that ends on a deliberate held frame is legitimate. But it must be a
  // decision somebody made, not a default.
  if (remaining < 0) {
    console.log(
      `    ${'!'.padEnd(14)} action ran ${spent.toFixed(1)}s but narration is only ` +
      `${seconds.toFixed(1)}s — the tail will be cut.`)
  } else if (act && spent < seconds * 0.5) {
    console.log(
      `    ${'?'.padEnd(14)} action used ${spent.toFixed(1)}s of ${seconds.toFixed(1)}s — the last ` +
      `${remaining.toFixed(1)}s holds a frozen aftermath while the narration is still ` +
      'talking. Pace it with `await hold(f)`.')
  }

  inPage(`async (page) => {
    await page.waitForTimeout(${Math.max(0, Math.ceil(remaining * 1000))})
    await page.evaluate(() => { document.title = 'FILMED' })
  }`)
  cli(['video-stop'])

  // The file appears when the browser flushes it, which is not instant.
  for (let attempt = 0; attempt < 40 && !existsSync(path); attempt += 1) sleep(250)
  if (!existsSync(path)) {
    throw new Error(`${path} was not written. A headless shell does this silently — see guard 4.`)
  }
}

// ── run ──────────────────────────────────────────────────────────────────────────────────
const timingPath = join(OUT, 'timing.json')
const timing = existsSync(timingPath) ? JSON.parse(readFileSync(timingPath, 'utf8')) : {}

const onlyFilm = process.argv[2]
const onlyScenes = process.argv[3] ? new Set(process.argv[3].split(',')) : null
const films = onlyFilm ? FILMS.filter(f => f.id === onlyFilm) : FILMS
if (!films.length) {
  throw new Error(`No film called \`${onlyFilm}\`. Known: ${FILMS.map(f => f.id).join(', ')}`)
}

for (const film of films) {
  const t = timing[film.id]
  if (!t) throw new Error(`No narration timing for ${film.id}. Run: node narrate.mjs ${film.id}`)

  const filmDir = join(OUT, film.id)
  const pagesDir = join(filmDir, 'pages')
  const videoDir = join(filmDir, 'video')
  mkdirSync(pagesDir, { recursive: true })
  mkdirSync(videoDir, { recursive: true })

  console.log(`\n${film.title}`)
  console.log('  preparing shots…')
  const prepared = await prepareFilm(film, { pagesDir })

  const server = await serveDir(pagesDir, RUNTIME.port)

  // Prove the socket answers before handing a URL to a browser: a silent bind failure shows
  // up as ERR_CONNECTION_REFUSED on the first scene of every film and reads as a browser bug.
  const firstPage = Object.values(prepared).find(p => p.page)?.page
  if (firstPage) {
    const probe = `http://127.0.0.1:${RUNTIME.port}/${firstPage}`
    const res = await fetch(probe).catch(e => ({ status: `fetch failed: ${e.message}` }))
    console.log(`  server    ${probe} -> ${res.status}`)
  }

  cli(['open', '--config', CONFIG], { tolerate: true })
  // The viewport and the video size are coupled: `inFrame()` measures against the viewport,
  // so a mismatch means it reports elements as visible that the recording does not contain.
  cli(['resize', String(VIDEO.width), String(VIDEO.height)])
  assertErrorChannelWorks()

  try {
    for (const scene of film.scenes) {
      if (onlyScenes && !onlyScenes.has(scene.id)) continue

      const seconds = t.scenes.find(s => s.id === scene.id)?.seconds
      if (!seconds) throw new Error(`${film.id}/${scene.id}: no measured narration length`)

      const shot = prepared[scene.id]
      if (!shot) throw new Error(`${film.id}/${scene.id}: no shot prepared`)

      const target = shot.url || `http://127.0.0.1:${RUNTIME.port}/${shot.page}`
      cli(['goto', target])

      // A positive assertion that the INTENDED screen rendered. Positive only: "the word I
      // expect is present" fails loudly on the wrong page, where "no error is visible"
      // passes on almost anything.
      if (shot.ready) {
        const seen = evaluate('() => document.body.innerText')
        if (!seen.includes(shot.ready)) {
          throw new Error(
            `${film.id}/${scene.id}: expected to see ${JSON.stringify(shot.ready)} at ${target}, ` +
            'but it is not on the page. Filming would have recorded the wrong screen.')
        }
      }

      const out = join(videoDir, `${scene.id}.webm`)
      record(out, seconds, { act: ACTS[film.id]?.[scene.id] ?? null, reveal: shot.reveal !== false })

      const dur = Number(execFileSync('ffprobe',
        ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
        { encoding: 'utf8' }).trim())
      const flag = dur < seconds * LIMITS.shortTakeRatio ? '  ⚠ short' : ''
      console.log(
        `  ${scene.id.padEnd(20)} narration ${seconds.toFixed(2).padStart(6)}s   ` +
        `video ${dur.toFixed(2).padStart(6)}s${flag}`)
    }
  } finally {
    cli(['close'], { tolerate: true })
    server.kill()
  }
}

console.log('\n  next: node assemble.mjs')
