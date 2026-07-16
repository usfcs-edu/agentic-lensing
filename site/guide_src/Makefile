# =============================================================================
# Build + verify the guide (site/docs/guide/).
#
#   make check     — the full gate: figures reproduce, numbers reproduce, lint
#   make figures   — regenerate all figures (must leave git clean)
#   make site      — build the mkdocs site strictly
#   make standalone— build guide/guide.pdf and guide/guide.html
#   make serve     — mkdocs serve for the browser QA pass
#   make all       — check + site + standalone
#
# One markdown source, three targets. See site/build_guide.py for why the
# PDF/HTML path needs a preprocessor at all.
# =============================================================================
PY      := $(HOME)/.venvs/lensjudge/bin/python
MKDOCS  := $(HOME)/.venvs/agentic-site/bin/mkdocs
REPO    := $(CURDIR)/../..
SRC     := $(CURDIR)

.PHONY: all check figures numbers lint site standalone serve clean

all: check site standalone

# The commit gate. Run this before every commit that touches the guide.
check: figures numbers lint
	@echo "== verifying figures are byte-reproducible =="
	@$(PY) $(SRC)/make_figures.py --check
	@git -C $(REPO) diff --quiet -- site/docs/guide/figures \
	  || (echo "FAIL: regenerating figures dirtied the tree."; \
	      echo "      _style.py must pin svg.hashsalt AND savefig(metadata={'Date':None})."; \
	      exit 1)
	@echo "== all gates pass =="

figures:
	@$(PY) $(SRC)/make_figures.py --pdf

# Every number in the prose must come from here and must reproduce.
# worked_examples --check pins our values against the repo's published ones;
# verify_numbers checks every <!-- check: --> tag in the prose against the code.
numbers:
	@$(PY) $(SRC)/worked_examples.py --check
	@$(PY) $(SRC)/worked_examples.py --emit
	@$(PY) $(SRC)/verify_numbers.py

# Catches the failures that are SILENT in the browser: raw TeX on the page,
# stacked light/dark images, unrenderable math, dead cross-refs.
lint:
	@$(PY) $(SRC)/lint_guide.py

site:
	@cd $(REPO) && python3 site/build_reports.py >/dev/null
	@$(MKDOCS) build -f $(REPO)/site/mkdocs.yml --strict

standalone:
	@cd $(REPO) && python3 site/build_guide.py --html --pdf

serve:
	@cd $(REPO) && python3 site/build_reports.py >/dev/null
	@$(MKDOCS) serve -f $(REPO)/site/mkdocs.yml

clean:
	@rm -rf $(REPO)/site/_site $(REPO)/guide/guide.pdf $(REPO)/guide/guide.html
