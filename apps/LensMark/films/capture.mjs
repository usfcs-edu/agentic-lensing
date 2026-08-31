/**
 * Run each target for real, once, and keep every byte it printed.
 *
 * The films are cut from THESE files. Nothing downstream re-runs a tool or edits its
 * output — `shots.mjs` only chooses which span of a capture a given scene shows. That is
 * the property worth protecting: if a command starts printing something awkward, the film
 * shows it, because there is no step in between that could quietly tidy it.
 *
 * Captures are written to `out/captures/<name>.txt` with a sidecar recording the command,
 * the exit code, the wall clock and the timestamp, so a claim in the narration can be
 * traced to the run that produced it.
 *
 *   node capture.mjs            # every target
 *   node capture.mjs <name>     # one
 *
 * Films made only of slides, cards or live web pages never need this.
 */
import { spawnSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { CAPTURE } from './film.config.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'out', 'captures')

const ANSI = new RegExp(String.fromCharCode(27) + '\\[[0-9;?]*[A-Za-z]', 'g')

function run(target) {
  const cwd = resolve(CAPTURE.root, target.dir ?? '.')
  const started = Date.now()
  const res = spawnSync(target.cmd, target.args ?? [], {
    cwd,
    encoding: 'utf8',
    maxBuffer: 128 * 1024 * 1024,
    // CAPTURE.env turns off theatrical sleeps and colour AT SOURCE. Stripping colour
    // afterwards works, but a tool that knows it is not on a terminal often prints better
    // output in the first place.
    env: { ...process.env, ...CAPTURE.env, ...(target.env ?? {}) },
  })
  if (res.error) throw new Error(`${target.name}: ${res.error.message} (in ${cwd})`)
  const seconds = (Date.now() - started) / 1000
  // stdout then stderr, in that order. Interleaving faithfully would need a pty; what
  // matters for a film is that nothing is discarded, and a tool's diagnostics are often
  // the part worth filming.
  const text = `${res.stdout ?? ''}${res.stderr ?? ''}`.replace(ANSI, '')
  return { text, code: res.status, seconds, cwd }
}

const only = process.argv[2]
const targets = only ? CAPTURE.targets.filter(t => t.name === only) : CAPTURE.targets

if (!CAPTURE.targets.length) {
  throw new Error(
    'No capture targets configured. Add them to CAPTURE.targets in film.config.mjs, e.g.\n' +
    "    { name: 'demo', dir: '.', cmd: 'bash', args: ['demo.sh'] }",
  )
}
if (!targets.length) {
  throw new Error(`No target \`${only}\`. Known: ${CAPTURE.targets.map(t => t.name).join(', ')}`)
}

mkdirSync(OUT, { recursive: true })
for (const t of targets) {
  process.stdout.write(`  ${t.name.padEnd(16)} running ${t.cmd} ${(t.args ?? []).join(' ')} … `)
  const r = run(t)
  writeFileSync(join(OUT, `${t.name}.txt`), r.text, 'utf8')
  writeFileSync(join(OUT, `${t.name}.json`), `${JSON.stringify({
    name: t.name,
    dir: t.dir ?? '.',
    cwd: r.cwd,
    command: `${t.cmd} ${(t.args ?? []).join(' ')}`,
    exitCode: r.code,
    seconds: Number(r.seconds.toFixed(2)),
    lines: r.text.split('\n').length,
    bytes: r.text.length,
    capturedAt: new Date().toISOString(),
  }, null, 2)}\n`, 'utf8')
  console.log(`exit ${r.code}  ${r.seconds.toFixed(1)}s  ${r.text.split('\n').length} lines`)
}
console.log(`\n  captures in ${OUT}`)
