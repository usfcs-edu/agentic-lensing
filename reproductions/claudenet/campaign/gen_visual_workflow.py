#!/usr/bin/env python3
"""gen_visual_workflow.py — generate the self-contained visual-judging Workflow
JS (data baked in, since the workflow sandbox has no filesystem). Emits
campaign/visual_workflow.js; launch with Workflow({scriptPath: that file}).

Phase A: 737 candidates chunked into batches of 8; each subagent Reads the 4
PNG views per candidate and grades against the Huang-VI rubric -> structured
grades. Phase B: every A/B is re-checked by a skeptic subagent (batch 4); the
grade holds only if the skeptic confirms >=B.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"
BASE = str(OUT / "png")

rubric = (ROOT / "campaign" / "prompts" / "rubric_visual.md").read_text()
skeptic = (Path("/home2/benson/git/agentic-lensing/reproductions/lensjudge/prompts/skeptic.md")
           .read_text())
items = json.load(open(OUT / "visual_input.json"))
compact = [{"row_id": d["row_id"], "p": round(d["p_final"], 4)} for d in items]
compact.sort(key=lambda d: d["row_id"])

js = r'''export const meta = {
  name: 'claudenet-visual-judge',
  description: 'Structured visual grading of the 737 ClaudeNet v2 sweep candidates (Read PNG views, Huang A/B/C/D rubric) + skeptic verification of A/B',
  phases: [
    { title: 'Grade', detail: 'batches of 8 candidates, subagents Read 4 views each' },
    { title: 'Skeptic', detail: 'adversarial re-check of every A/B' },
  ],
}

const RUBRIC = ''' + json.dumps(rubric) + r''';
const SKEPTIC = ''' + json.dumps(skeptic) + r''';
const BASE = ''' + json.dumps(BASE) + r''';
const ITEMS = ''' + json.dumps(compact, separators=(",", ":")) + r''';

const VIEWS = ["full", "zoom", "residual", "highcontrast"];
function paths(rid) { return VIEWS.map(v => `${BASE}/${rid}/${v}.png`); }

const GRADE = {
  type: "object",
  properties: {
    row_id: { type: "string" },
    grade: { type: "string", enum: ["A", "B", "C", "D"] },
    criteria: { type: "object", properties: {
      blue_source: { type: "number" }, low_surface_brightness: { type: "number" },
      curvature: { type: "number" }, counter_images: { type: "number" },
      arc_morphology: { type: "number" } },
      required: ["blue_source","low_surface_brightness","curvature","counter_images","arc_morphology"] },
    p_lens: { type: "number" }, confidence: { type: "number" },
    contaminant: { type: ["string","null"] },
    escalate_to_human: { type: "boolean" },
    rationale: { type: "string" },
  },
  required: ["row_id","grade","criteria","p_lens","confidence","contaminant","escalate_to_human","rationale"],
};
const BATCH_SCHEMA = { type: "object", properties: { grades: { type: "array", items: GRADE } }, required: ["grades"] };
const SKEPTIC_SCHEMA = { type: "object", properties: {
  row_id: { type: "string" },
  skeptic_grade: { type: "string", enum: ["A","B","C","D"] },
  skeptic_confirms: { type: "boolean" },
  skeptic_contaminant: { type: ["string","null"] },
  skeptic_rationale: { type: "string" },
}, required: ["row_id","skeptic_grade","skeptic_confirms","skeptic_contaminant","skeptic_rationale"] };

function chunk(a, n) { const o=[]; for (let i=0;i<a.length;i+=n) o.push(a.slice(i,i+n)); return o; }

phase('Grade')
const batches = chunk(ITEMS, 8);
log(`grading ${ITEMS.length} candidates in ${batches.length} batches of 8`);
const graded = await parallel(batches.map((b, bi) => () => {
  const lines = b.map(c => {
    const [full, zoom, residual, highcontrast] = paths(c.row_id);
    return `- row_id=${c.row_id}  p_final=${c.p}\n    full: ${full}\n    zoom: ${zoom}\n    residual: ${residual}\n    highcontrast: ${highcontrast}`;
  }).join("\n");
  const prompt = `${RUBRIC}\n\n# Your batch (${b.length} candidates)\nFor EACH candidate below, use the Read tool on ALL FOUR PNG view paths, then grade it. Return {"grades": [...]} with one object per candidate, row_id copied exactly.\n\n${lines}`;
  return agent(prompt, { label: `grade:b${bi}`, phase: 'Grade', schema: BATCH_SCHEMA })
    .then(r => (r && r.grades) ? r.grades : [])
    .catch(() => []);
}));
const allGrades = graded.flat();
log(`first-pass grades: ${allGrades.length}/${ITEMS.length}`);

phase('Skeptic')
const ab = allGrades.filter(g => g.grade === "A" || g.grade === "B");
log(`skeptic re-check on ${ab.length} A/B candidates`);
const verdicts = await parallel(ab.map((g) => () => {
  const [full, zoom, residual, highcontrast] = paths(g.row_id);
  const prompt = `${SKEPTIC}\n\n${RUBRIC}\n\n# Candidate to refute\nrow_id=${g.row_id}. The first-pass grader gave grade=${g.grade}, p_lens=${g.p_lens}, rationale="${g.rationale}". Read ALL FOUR views (full: ${full}\nzoom: ${zoom}\nresidual: ${residual}\nhighcontrast: ${highcontrast}) and decide: does the lens evidence survive skeptical scrutiny? Return your own grade; skeptic_confirms=true only if it stays >=B (A or B).`;
  return agent(prompt, { label: `skeptic:${g.row_id}`, phase: 'Skeptic', schema: SKEPTIC_SCHEMA })
    .then(v => v || null).catch(() => null);
}));

return { grades: allGrades, skeptic: verdicts.filter(Boolean) };
'''

dest = ROOT / "campaign" / "visual_workflow.js"
dest.write_text(js)
print(f"[gen] wrote {dest} ({len(js):,} bytes; {len(compact)} candidates baked in)")
