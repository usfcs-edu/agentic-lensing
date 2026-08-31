/**
 * A loopback static server for the generated shot pages, run as its OWN PROCESS.
 *
 * ─── WHY NOT file:// ──────────────────────────────────────────────────────────────────
 * playwright-cli refuses the `file:` protocol outright ("Access to 'file:' protocol is
 * blocked"). Older builds allowed it, which is why the pipeline this generalizes navigated
 * to file URLs for years and then stopped working. Checked, not assumed.
 *
 * ─── WHY A SEPARATE PROCESS ───────────────────────────────────────────────────────────
 * `record.mjs` drives the browser with `execFileSync('playwright-cli', …)`, which blocks
 * its event loop for the whole duration of every call — so a server sharing that loop
 * cannot answer the very navigation it was started to serve. Node's own `fetch` still
 * succeeds, because it runs in the gaps between calls, and Chromium times out after sixty
 * seconds. The symptom reads as a browser fault. It is a deadlock, and moving the socket to
 * another process is the fix.
 *
 *   node serve.mjs <pagesRoot> <port> [assetsRoot]   # prints "listening" once bound
 *
 * `assetsRoot` is mounted at /assets and holds the bundled font. Serving it from the same
 * origin matters: a woff2 fetched cross-origin without CORS is dropped silently, and the
 * page then renders in a fallback font that looks almost right.
 */
import { createServer } from 'node:http'
import { readFileSync, statSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'

const root = resolve(process.argv[2])
const port = Number(process.argv[3] || 8099)
const assets = process.argv[4] ? resolve(process.argv[4]) : null

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml',
  '.webp': 'image/webp', '.wav': 'audio/wav',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf', '.otf': 'font/otf',
}

createServer((req, res) => {
  const rel = decodeURIComponent((req.url || '/').split('?')[0]).replace(/^\/+/, '')

  let base = root
  let path = rel
  if (assets && (rel === 'assets' || rel.startsWith('assets/'))) {
    base = assets
    path = rel.slice('assets'.length).replace(/^\/+/, '')
  }

  let file = join(base, path || 'index.html')
  try { if (statSync(file).isDirectory()) file = join(file, 'index.html') } catch { /* not a dir */ }

  // Traversal guard. `resolve` above means both roots are absolute and normalized, so a
  // `../` in the request cannot climb out.
  if (!file.startsWith(base)) { res.writeHead(403).end('outside root'); return }

  try {
    const body = readFileSync(file)
    res.writeHead(200, {
      'content-type': TYPES[extname(file)] || 'application/octet-stream',
      // Every take must see the page the recorder just wrote, not the one before it.
      'cache-control': 'no-store',
    })
    res.end(body)
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' }).end(`not found: ${rel}`)
  }
}).listen(port, '127.0.0.1', () => console.log('listening'))
