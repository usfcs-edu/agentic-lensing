/**
 * Speak one line in several voices, so the voice is CHOSEN rather than defaulted into.
 *
 * ─── WHY THIS EXISTS ──────────────────────────────────────────────────────────────────
 *
 * `film.config.mjs` has always said "audition before committing", and `speak.py` has always
 * carried a per-item `voice` override that exists for nothing else. Until this file, no
 * caller ever sent one: the advice was a comment nobody could act on, and every film
 * shipped in whichever voice the template happened to contain.
 *
 * ─── THE LENGTH OF THE FILM IS A PROPERTY OF THE VOICE ────────────────────────────────
 *
 * The same sentence at the same speed measured 8.3s in one voice and 10.5s in another — a
 * 26% spread. In a pipeline that paces every picture to the measured narration that is not
 * a curiosity, it is the difference between a three-minute film and a four-minute one,
 * decided by one string in `film.config.mjs`. The table below prints that spread, and when
 * the line came from your own script it prints what the spread does to the finished
 * runtime. It is also why a voice is expensive to change late: every scene's length is
 * measured from it, so swapping voices after a film is cut re-times every picture in it.
 *
 * ─── WHY VOICES ARE GROUPED BY ACCENT RATHER THAN RENDERED ONE AT A TIME ──────────────
 *
 * `speak.py` takes `voice` per ITEM but `speed` and `langCode` per REQUEST. `langCode`
 * picks the phonemizer and must agree with the voice prefix — `af_`/`am_` are American
 * ('a'), `bf_`/`bm_` British ('b') — and a mismatch reads with the wrong vowel set, which
 * sounds like a bad take rather than a bug. So voices are BUCKETED by their own prefix and
 * one request is sent per bucket: six voices spanning both accents is two python processes,
 * not six, and all twenty-eight English voices is still two. The langCode is derived from
 * the name and never accepted from the caller, so it cannot be got wrong.
 *
 * ─── AN AUDITION IS NOT A TAKE ────────────────────────────────────────────────────────
 *
 * Nothing here writes `out/timing.json`. That is structural rather than careful: this file
 * does not import `writeFileSync` at all. Auditions land in `out/_auditions/` and are
 * overwritten freely.
 *
 *   node audition.mjs                                # six voices, the first line of your script
 *   node audition.mjs af_heart,bm_lewis              # your shortlist
 *   node audition.mjs default "Say this instead."    # your line
 *   node audition.mjs all                            # every English voice, still two requests
 *   node audition.mjs list                           # what is really in voices-v1.0.bin
 *
 * `FILM_VOICE_SPEED` overrides `VOICE.speed` for one run. Speed moves the durations too,
 * and it is per-request, so one value applies to the whole audition.
 */
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { LIMITS, RUNTIME, VIDEO, VOICE } from './film.config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const SPEAK = join(HERE, 'speak.py')
// `_` prefixed like `out/_program.js` and `out/_boom.js`: generated, never a deliverable.
const OUT = join(HERE, 'out', '_auditions')

const LANG_NAME = { a: 'en-us', b: 'en-gb' }

/**
 * Six, not twenty-eight.
 *
 * A shortlist you can sit through in a couple of minutes, spanning both accents and both
 * genders, with the template's own default first so everything else is heard against it.
 * `all` exists for completeness, but twenty-eight voices is not an audition, it is an
 * afternoon.
 */
const DEFAULT_VOICES = ['af_heart', 'af_bella', 'af_nicole', 'am_michael', 'bf_emma', 'bm_george']

/** Used only when `script.mjs` has nothing to offer. Long enough to hear the prosody. */
const FALLBACK_TEXT =
  'The words come first, and the pictures are paced to them. This sentence was synthesised ' +
  'on this machine, measured, and nothing about it was trimmed to fit.'

/** Seconds of audio in a file, read from the file rather than trusted from the sidecar. */
function durationOf(path) {
  return Number(
    execFileSync(
      'ffprobe',
      ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
      { encoding: 'utf8' },
    ).trim(),
  )
}

/**
 * Mean volume in dBFS.
 *
 * `volumedetect` reports on STDERR, like everything ffmpeg says about a file, so this reads
 * BOTH streams. Reading only stdout returns an empty string, the regex misses, and every
 * voice in the table is reported as silence. See narrate.mjs, where the same mistake once
 * refused a set of perfectly good takes.
 */
function meanVolume(path) {
  const run = spawnSync(
    'ffmpeg',
    ['-hide_banner', '-i', path, '-af', 'volumedetect', '-f', 'null', '-'],
    { encoding: 'utf8' },
  )
  const said = `${run.stdout ?? ''}${run.stderr ?? ''}`
  return Number(/mean_volume:\s*(-?\d+(?:\.\d+)?) dB/.exec(said)?.[1] ?? NaN)
}

/** The interpreter check, in the one place that would otherwise fail confusingly. */
function requirePython() {
  if (existsSync(RUNTIME.python)) return
  throw new Error(
    `No Python interpreter with kokoro-onnx at ${RUNTIME.python}.\n\n` +
      (process.env.FILM_PYTHON
        ? 'That path came from FILM_PYTHON, so the variable is set and the path is wrong.\n'
        : `Build one:\n\n    bash <skill>/scripts/setup.sh --dir ${HERE}\n\n`),
  )
}

/**
 * The voice names actually in the pack.
 *
 * Asked of `speak.py`, which already owns that file's format — a second reader of it here
 * would be a second thing to keep true. `--list-voices` does not load the model, so this
 * costs a moment and lets a typo be refused BEFORE 337 MB of weights are read.
 *
 * Returns null rather than throwing: auditioning must still work when the list cannot be
 * had, it just stops being checked.
 */
function voicesInPack() {
  try {
    const run = spawnSync(RUNTIME.python, [SPEAK, '--list-voices'], {
      encoding: 'utf8',
      env: { ...process.env, FILM_MODELS: RUNTIME.models, TOKENIZERS_PARALLELISM: 'false' },
    })
    for (const line of `${run.stdout ?? ''}`.split('\n')) {
      if (!line.startsWith('{')) continue
      const row = JSON.parse(line)
      if (Array.isArray(row.voices)) return row.voices
    }
    return null
  } catch {
    return null
  }
}

/**
 * The langCode a voice REQUIRES, taken from its own name.
 *
 * Kokoro names carry language and gender: `af_` American female, `bm_` British male.
 * `speak.py` maps 'a' -> en-us and 'b' -> en-gb with a `.get(…, "en-us")` DEFAULT, so a
 * Japanese voice would not error — it would be phonemized as English and merely sound
 * wrong, with nothing anywhere reporting a problem. That silence is why this refuses
 * rather than guessing.
 */
function langOf(voice) {
  const code = voice[0]
  if (code !== 'a' && code !== 'b') {
    throw new Error(
      `\`${voice}\` is not a voice this pipeline can speak.\n\n` +
        "  speak.py maps langCode 'a' -> en-us and 'b' -> en-gb and nothing else, so only the\n" +
        '  af_ am_ bf_ bm_ voices are reachable. The rest of the pack is Spanish, French,\n' +
        '  Hindi, Italian, Japanese, Portuguese and Mandarin; to reach those, widen the map in\n' +
        '  speak.py. Left as it is, that voice would be read as English and sound like a bad\n' +
        '  take rather than an unsupported language.\n\n' +
        '  What you can use:  node audition.mjs list',
    )
  }
  return code
}

/**
 * One request per accent — the correctness requirement and the fast path at once.
 *
 * Voices sharing an accent must travel together and voices that do not must not, because
 * `langCode` is per-request. That the model load is then paid once per accent rather than
 * once per voice is a consequence, not the reason.
 */
function groupByLang(voices) {
  const groups = new Map()
  for (const voice of voices) {
    const lang = langOf(voice)
    if (!groups.has(lang)) groups.set(lang, [])
    groups.get(lang).push(voice)
  }
  return groups
}

/** One `speak.py` invocation: one accent, many voices, each item naming its own. */
function render(items, lang, speed) {
  const request = JSON.stringify({
    // Required by the protocol, and the value each item's `voice` overrides. Every item
    // here names one, so this only ever labels the request.
    voice: items[0].voice,
    speed,
    langCode: lang,
    items,
  })

  const run = spawnSync(RUNTIME.python, [SPEAK], {
    input: request,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, FILM_MODELS: RUNTIME.models, TOKENIZERS_PARALLELISM: 'false' },
    stdio: ['pipe', 'pipe', 'inherit'],
  })
  if (run.error) throw run.error

  const rendered = new Map()
  for (const line of `${run.stdout ?? ''}`.split('\n')) {
    if (!line.startsWith('{')) continue
    const row = JSON.parse(line)
    // speak.py prints its error line and THEN exits non-zero, so the useful sentence is in
    // the output rather than the status. Read it before falling back to the status.
    if (row.error !== undefined) throw new Error(`${row.id}: ${row.error}`)
    rendered.set(row.id, row)
  }
  if (run.status !== 0) throw new Error(`speak.py exited ${run.status} without saying why.`)
  return rendered
}

/**
 * The line to audition, preferring your OWN first scene.
 *
 * Auditioning a stock sentence tells you which voice reads a stock sentence well. The point
 * is the film, so the first scene of the first film in `script.mjs` wins when there is one,
 * and the durations below become the real durations of a real scene.
 *
 * DYNAMICALLY imported and swallowed on failure, deliberately. `narrate.mjs` imports
 * `script.mjs` at the top because it cannot do its job without it. Auditioning happens
 * BEFORE that file exists and while it is half-written, and a static import would make a
 * stray comma the reason you cannot listen to a voice.
 */
async function scriptLine() {
  try {
    const { FILMS } = await import('./script.mjs')
    const film = FILMS?.[0]
    const scene = film?.scenes?.[0]
    if (typeof scene?.text !== 'string' || scene.text.trim() === '') return null
    return {
      text: scene.text.trim(),
      from: `script.mjs — ${film.id}/${scene.id}`,
      filmId: film.id,
      filmScenes: film.scenes.length,
      filmWords: film.scenes.reduce((n, s) => n + s.text.trim().split(/\s+/).length, 0),
    }
  } catch {
    return null
  }
}

const words = text => text.trim().split(/\s+/).length
const clock = s => `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, '0')}s`

// ── arguments ─────────────────────────────────────────────────────────────────────────────
const [voicesArg, textArg] = process.argv.slice(2)

requirePython()
const pack = voicesInPack()

if (voicesArg === 'list') {
  if (!pack) {
    throw new Error(
      `Cannot read the voice pack under ${RUNTIME.models}.\n\n` +
        '  Run scripts/setup.sh, or point FILM_MODELS at a directory holding it.',
    )
  }
  const usable = pack.filter(v => v[0] === 'a' || v[0] === 'b')
  console.log(`${pack.length} voices in the pack; ${usable.length} of them speakable here.\n`)
  for (const lang of ['a', 'b']) {
    console.log(`  ${LANG_NAME[lang]}   ${usable.filter(v => v[0] === lang).join(' ')}`)
  }
  const rest = pack.filter(v => v[0] !== 'a' && v[0] !== 'b')
  console.log(
    `\n  Not reachable (${rest.length}): ${rest.join(' ')}\n` +
      "  speak.py maps only 'a' -> en-us and 'b' -> en-gb. Passing one of these would not\n" +
      '  error — it would be phonemized as English. Widen that map first.',
  )
  process.exit(0)
}

const speed = Number(process.env.FILM_VOICE_SPEED || VOICE.speed)
if (!Number.isFinite(speed) || speed <= 0) {
  throw new Error(`FILM_VOICE_SPEED=${process.env.FILM_VOICE_SPEED} is not a positive number.`)
}

let requested
if (!voicesArg || voicesArg === 'default') {
  requested = DEFAULT_VOICES
} else if (voicesArg === 'all') {
  if (!pack) {
    throw new Error(
      '`all` needs to read the voice pack, and cannot.\n\n' +
        '  Name the voices instead:  node audition.mjs af_heart,bm_george',
    )
  }
  requested = pack.filter(v => v[0] === 'a' || v[0] === 'b')
} else {
  requested = voicesArg.split(',').map(s => s.trim()).filter(Boolean)
}
requested = [...new Set(requested)]
if (requested.length === 0) throw new Error('No voices to audition.')

// Checked against the pack BEFORE a model load, so `bm_fabel` costs a moment rather than a
// minute. Skipped silently when the pack could not be listed.
if (pack) {
  const unknown = requested.filter(v => !pack.includes(v))
  if (unknown.length > 0) {
    throw new Error(
      `Not in the voice pack: ${unknown.join(', ')}.\n\n` +
        '  speak.py would have failed on these too, but only after loading the model.\n' +
        '  See what exists:  node audition.mjs list',
    )
  }
}

const source = textArg
  ? { text: textArg.trim(), from: 'the command line' }
  : (await scriptLine()) ?? { text: FALLBACK_TEXT, from: 'the built-in line (no readable script.mjs)' }

// ── render ────────────────────────────────────────────────────────────────────────────────
const groups = groupByLang(requested)
mkdirSync(OUT, { recursive: true })

console.log(`Auditioning ${requested.length} voice(s) at ${speed}×, in ${groups.size} request(s):\n`)
for (const [lang, voices] of groups) console.log(`  ${LANG_NAME[lang]}   ${voices.join(' ')}`)
console.log(`\n  text    ${source.from}`)
console.log(`  “${source.text}”\n`)

const results = new Map()
for (const [lang, voices] of groups) {
  const items = voices.map(voice => ({
    id: voice,
    text: source.text,
    path: join(OUT, `${voice}.wav`),
    // The per-item override speak.py documents as existing "only for auditioning".
    voice,
  }))
  for (const [id, row] of render(items, lang, speed)) results.set(id, row)
}

// ── measure ───────────────────────────────────────────────────────────────────────────────
const rows = []
for (const voice of requested) {
  const path = join(OUT, `${voice}.wav`)
  if (!results.has(voice) || !existsSync(path)) {
    throw new Error(`${voice}: the sidecar reported success and wrote no file.`)
  }
  const db = meanVolume(path)
  rows.push({
    voice,
    lang: langOf(voice),
    seconds: durationOf(path),
    db,
    quiet: !Number.isFinite(db) || db < LIMITS.silenceDb,
  })
}

// Flagged per row rather than fatal, unlike narrate.mjs: one dead voice should not cost you
// the other five. Every voice silent is the model-loaded-and-emitted-nothing failure, and
// that still stops everything.
if (rows.every(r => r.quiet)) {
  throw new Error(
    `Every voice measured below ${LIMITS.silenceDb} dBFS. A model can load, report success ` +
      'and emit nothing; that is what this check is for.',
  )
}

const fastest = Math.min(...rows.map(r => r.seconds))
const slowest = Math.max(...rows.map(r => r.seconds))

console.log(
  `  ${'voice'.padEnd(14)}${'lang'.padEnd(8)}${'seconds'.padStart(8)}` +
    `${'vs fastest'.padStart(13)}${'mean dBFS'.padStart(12)}`,
)
console.log(`  ${'—'.repeat(55)}`)
for (const r of rows) {
  const rel = r.seconds === fastest ? '—' : `+${((r.seconds / fastest - 1) * 100).toFixed(1)}%`
  console.log(
    `  ${r.voice.padEnd(14)}${LANG_NAME[r.lang].padEnd(8)}${r.seconds.toFixed(2).padStart(7)}s` +
      `${rel.padStart(13)}${(Number.isFinite(r.db) ? r.db.toFixed(1) : 'n/a').padStart(12)}` +
      (r.quiet ? '   ← silence' : ''),
  )
}

// ── the point ─────────────────────────────────────────────────────────────────────────────
console.log(
  `\n  ${fastest.toFixed(2)}s to ${slowest.toFixed(2)}s — a ` +
    `${((slowest / fastest - 1) * 100).toFixed(0)}% spread on one sentence at one speed.\n` +
    '  Every picture is paced to the measured narration, so the voice decides how long\n' +
    '  the finished film runs.',
)

if (source.filmWords) {
  const scale = source.filmWords / words(source.text)
  const air = source.filmScenes * VIDEO.tailSeconds
  console.log(
    `\n  Scaled to ${source.filmId}'s ${source.filmWords} words over ${source.filmScenes} scenes: roughly ` +
      `${clock(fastest * scale + air)} against ${clock(slowest * scale + air)}\n` +
      '  of finished film. An estimate — it assumes each voice holds its words-per-second\n' +
      '  across the whole script — but the direction is real.',
  )
}

console.log(`\n  Listen, in the order above:\n`)
console.log(
  process.platform === 'darwin'
    ? `    for v in ${rows.map(r => r.voice).join(' ')}; do echo "  $v"; afplay ${join(OUT, '$v.wav')}; done`
    : `    for v in ${rows.map(r => r.voice).join(' ')}; do echo "  $v"; ffplay -v error -autoexit -nodisp ${join(OUT, '$v.wav')}; done`,
)
console.log(`\n  Files: ${OUT}`)
console.log(
  '\n  To commit a choice, edit BOTH fields in film.config.mjs together — langCode always\n' +
    "  follows the voice's first letter — or override them for one run:\n\n" +
    '      FILM_VOICE=<voice> FILM_LANG=<a|b> node narrate.mjs\n\n' +
    "  Then re-run `node narrate.mjs`: every scene's length is measured from the voice, and\n" +
    '  has just changed.',
)
