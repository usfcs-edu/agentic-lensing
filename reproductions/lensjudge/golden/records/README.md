# `golden/records/` — raw grading-tool events

What belongs here: the JSONL files the grading tool (`kits/<kit_id>/grade.html`, or its
`serve.py`) emits — one file per kit and browser session, named
`events_<kit_id>_<session_id>.jsonl`, one JSON object per commit with exactly the 14
`GradeEvent` keys (`schema.py`): `kit_id, manifest_sha, item_id, presentation_index,
session_id, grader_id, score_1_4, confidence_lmh, seconds, revision, flag, timestamp, ua,
tool_version`. The auto-downloads overlap (every 20 commits, and the full store on export),
so the same event may appear in several files; `collect.py` de-duplicates exact copies and
aborts on conflicting ones.

These files are the primary record of the campaign and are **tracked** (they are small).
Copy every file Xiaosheng sends back here unchanged, then run

    python lensjudge/golden/collect.py --kit-id <kit_id> --events "lensjudge/golden/records/events_<kit_id>_*.jsonl"

Never put anything else in this directory: no synthetic or dry-run events (the integration
dry run wrote `events_jwst_lite_v1_DRYRUN.jsonl` here and deleted it again), no edited
copies, no notes. An event file with an extra key is refused by the collector by design.
