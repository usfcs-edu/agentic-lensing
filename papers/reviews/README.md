# Reviews — three LLM-in-astronomy papers (August 2026)

Internal reviews prepared for the Strong Lensing Team. Each document exists as `.md`
(source), `.docx`, and `.pdf`.

| Document | Pages | Paper reviewed |
|---|---|---|
| `00-overview-three-papers` | 2 | Combined reading of all three |
| `01-stoppa-2025-llm-transient-classification` | 3 | Stoppa et al. 2025, *Nature Astronomy* 9, 1869 — LLM real/bogus classification with textual rationales |
| `02-astroalertbench-2026` | 3 | Chen et al. 2026, arXiv:2605.05573 — AstroAlertBench: accuracy, reasoning and honesty of multimodal LLMs |
| `03-alerce-text-to-sql-2026` | 3 | Estévez et al. 2026, arXiv:2606.18108 — the ALeRCE text-to-SQL system |

Each review carries a summary, main contributions, a critical assessment, and a section
relating the paper to the agentic lensing programme in this repo (JWST agentic search,
LensJudge, the human-parity programme, the evidence-first regrade, and LensMark).

Source PDFs are the siblings of this directory in `papers/`.

## Rebuilding

```bash
./build.sh            # rebuild every .docx and .pdf from the .md sources
./build.sh 02-astroalertbench-2026.md   # or just one
```

`build.sh` uses pandoc with `.reference.docx` (11 pt Charter, 1 in margins) for Word and
xelatex with `.header.tex` for PDF. Both are tuned so each review lands on its page target.
