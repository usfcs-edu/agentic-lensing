/**
 * Render the narration, measure it, and refuse anything that is silent.
 *
 * ─── THIS RUNS FIRST, AND THAT IS THE WHOLE DESIGN ────────────────────────────────────
 *
 * The words come first and the pictures are paced to them. Narration is synthesized here
 * and MEASURED into `out/timing.json`; only then does `record.mjs` decide how long each
 * scene stays on screen. Nothing is trimmed to fit afterwards.
 *
 * Invert that and you get the usual demo video: footage shot to a guess, narration written
 * to the footage, and a voice describing a screen that changed four seconds ago.
 *
 * ─── WHY EVERY SCENE IS MEASURED FOR LOUDNESS ─────────────────────────────────────────
 *
 * A model can load, report success, and emit silence. `volumedetect` on every rendered
 * scene is the only thing standing between that and a film that plays for four minutes
 * saying nothing — and which fails at no other point in the pipeline.
 *
 *   node narrate.mjs            # every film
 *   node narrate.mjs <filmId>   # one
 */
import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { FILMS } from './script.mjs'
import { LIMITS, RUNTIME, VOICE } from './film.config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const SPEAK = join(HERE, 'speak.py')
const OUT = join(HERE, 'out')

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
 * BOTH streams. Reading only stdout returns an empty string, the regex misses, the caller
 * gets NaN, and the silence check below then refuses a set of perfectly good takes. That is
 * a real failure from the pipeline this generalizes; reading both streams is how it stays
 * fixed.
 */
function meanVolume(path) {
  const run = spawnSync(
    'ffmpeg',
    // `-f null -` rather than `-f null /dev/null`: same thing, one fewer platform assumption.
    ['-hide_banner', '-i', path, '-af', 'volumedetect', '-f', 'null', '-'],
    { encoding: 'utf8' },
  )
  const said = `${run.stdout ?? ''}${run.stderr ?? ''}`
  return Number(/mean_volume:\s*(-?\d+(?:\.\d+)?) dB/.exec(said)?.[1] ?? NaN)
}

/**
 * Every scene of every film in ONE python process.
 *
 * Loading the model costs about a second, and the sidecar is started once per RUN rather
 * than once per scene so a twenty-scene film pays it once.
 */
function render(items) {
  if (!existsSync(RUNTIME.python)) {
    throw new Error(
      `No Python interpreter with kokoro-onnx at ${RUNTIME.python}.\n\n` +
        (process.env.FILM_PYTHON
          ? 'That path came from FILM_PYTHON, so the variable is set and the path is wrong.\n'
          : 'Build one:\n\n    bash <skill>/scripts/setup.sh --dir ' + HERE + '\n\n') +
        'There is no `say` fallback on purpose: `say` is concatenative and lands every ' +
        'sentence with the same shape. Fix the interpreter rather than shipping a lesser voice.',
    )
  }

  const request = JSON.stringify({
    voice: VOICE.voice,
    speed: VOICE.speed,
    langCode: VOICE.langCode,
    items,
  })

  const out = execFileSync(RUNTIME.python, [SPEAK], {
    input: request,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, FILM_MODELS: RUNTIME.models, TOKENIZERS_PARALLELISM: 'false' },
    stdio: ['pipe', 'pipe', 'inherit'],
  })

  const rendered = new Map()
  for (const line of out.split('\n')) {
    if (!line.startsWith('{')) continue
    const row = JSON.parse(line)
    if (row.error !== undefined) throw new Error(`${row.id}: ${row.error}`)
    rendered.set(row.id, row)
  }
  return rendered
}

const only = process.argv[2]
const films = only ? FILMS.filter(film => film.id === only) : FILMS
if (films.length === 0) {
  throw new Error(`No film called \`${only}\`. Known: ${FILMS.map(f => f.id).join(', ')}`)
}

const items = []
for (const film of films) {
  mkdirSync(join(OUT, film.id, 'audio'), { recursive: true })
  for (const scene of film.scenes) {
    items.push({
      id: `${film.id}/${scene.id}`,
      text: scene.text,
      path: join(OUT, film.id, 'audio', `${scene.id}.wav`),
    })
  }
}

console.log(`Rendering ${items.length} scene(s) with ${VOICE.voice} at ${VOICE.speed}×…\n`)
render(items)

// MERGED into whatever is already there, never replaced.
//
// `narrate.mjs one-film` rendering only that film and then writing a timing file containing
// only that film is how the other films lose their timings and the recorder refuses to run.
// A partial render must not destroy the record of a complete one.
const timingPath = join(OUT, 'timing.json')
const timing = existsSync(timingPath) ? JSON.parse(readFileSync(timingPath, 'utf8')) : {}
let grandTotal = 0

for (const film of films) {
  const scenes = []
  let total = 0

  for (const scene of film.scenes) {
    const path = join(OUT, film.id, 'audio', `${scene.id}.wav`)
    if (!existsSync(path)) {
      throw new Error(`${scene.id}: the sidecar reported success and wrote no file.`)
    }

    const seconds = durationOf(path)
    const db = meanVolume(path)
    if (!Number.isFinite(db) || db < LIMITS.silenceDb) {
      throw new Error(
        `${scene.id}: mean volume ${db} dBFS is silence. A model can load, report success ` +
          'and emit nothing; that is what this check is for.',
      )
    }

    scenes.push({
      id: scene.id,
      seconds: Number(seconds.toFixed(2)),
      meanDb: db,
      words: scene.text.trim().split(/\s+/).length,
    })
    total += seconds
    console.log(`  ${scene.id.padEnd(20)} ${seconds.toFixed(2).padStart(6)}s  ${String(db).padStart(7)} dBFS`)
  }

  timing[film.id] = { ...VOICE, title: film.title, total: Number(total.toFixed(2)), scenes }
  grandTotal += total
  console.log(`  ${'—'.repeat(48)}\n  ${film.id.padEnd(20)} ${total.toFixed(2).padStart(6)}s total\n`)
}

writeFileSync(timingPath, `${JSON.stringify(timing, null, 2)}\n`, 'utf8')
console.log(`${grandTotal.toFixed(1)}s of narration across ${films.length} film(s).`)
console.log(`  timing  ${timingPath}`)
console.log('\n  next: node record.mjs')
