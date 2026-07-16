# =============================================================================
# Build + verify the guides (site/docs/<guide>/).
#
#   make check      — the full gate for every guide
#   make check-primer — one guide only
#   make site       — build the mkdocs site strictly (covers both)
#   make standalone — build guide/<name>.pdf and guide/<name>.html for each
#   make serve      — mkdocs serve for the browser QA pass
#
# Adding a book is one word here plus an entry in guide_src/guides.py.
# =============================================================================
PY      := $(HOME)/.venvs/lensjudge/bin/python
MKDOCS  := $(HOME)/.venvs/agentic-site/bin/mkdocs
REPO    := $(CURDIR)/../..
SRC     := $(CURDIR)

GUIDES  := guide primer

.PHONY: all check site standalone serve clean

all: check site standalone

# The commit gate. Run before every commit that touches a guide.
check: $(addprefix check-,$(GUIDES))
	@echo "== all gates pass =="

site:
	@cd $(REPO) && python3 site/build_reports.py >/dev/null
	@$(MKDOCS) build -f $(REPO)/site/mkdocs.yml --strict

standalone: $(addprefix standalone-,$(GUIDES))

# =============================================================================
# Per-guide rules are GENERATED — one set per entry in $(GUIDES) — rather than
# written once as pattern rules (`check-%:`). That is not a style preference.
#
# GNU make NEVER applies a PATTERN rule to a target declared .PHONY. So the
# obvious spelling, which is what this file used to say:
#
#     .PHONY: check-guide check-primer
#     check-%:
#             ...the whole gate...
#
# matches NOTHING. `make check` then finds no recipe for its prerequisites,
# decides they are already satisfied, and prints "== all gates pass ==" having
# run zero figures, zero examples, zero numbers and zero lint. It did exactly
# that here: a green commit gate that checked nothing at all, which is strictly
# worse than having no gate. Reproduced minimally on GNU Make 3.81 (macOS):
# identical Makefile, .PHONY line removed -> the recipe fires; restored -> silence.
#
# Explicit rules are immune to this, and .PHONY still works on them, so we get
# both halves. Adding a book remains one word in GUIDES above.
# =============================================================================
define GUIDE_RULES
.PHONY: check-$(1) standalone-$(1)

# Three stages, in this order: regenerate -> assert byte-identical -> assert the
# tree is clean. The last one is what proves the committed figures are
# reproducible rather than merely present.
check-$(1):
	@echo "== $(1) =="
	@$$(PY) $$(SRC)/make_figures.py --guide $(1) --pdf
	@$$(PY) $$(SRC)/make_figures.py --guide $(1) --check
	@git -C $$(REPO) diff --quiet -- site/docs/$(1)/figures \
	  || (echo "FAIL: regenerating $(1) figures dirtied the tree."; \
	      echo "      _style.py must pin svg.hashsalt AND savefig(metadata={'Date':None})."; \
	      exit 1)
	@$$(PY) $$(SRC)/make_examples.py --guide $(1) --check
	@$$(PY) $$(SRC)/make_examples.py --guide $(1) --emit
	@$$(PY) $$(SRC)/verify_numbers.py --guide $(1)
	@$$(PY) $$(SRC)/lint_guide.py --guide $(1)

standalone-$(1):
	@cd $$(REPO) && python3 site/build_guide.py --guide $(1) --html --pdf
endef

$(foreach g,$(GUIDES),$(eval $(call GUIDE_RULES,$(g))))

serve:
	@cd $(REPO) && python3 site/build_reports.py >/dev/null
	@$(MKDOCS) serve -f $(REPO)/site/mkdocs.yml

clean:
	@rm -rf $(REPO)/site/_site
	@rm -f $(foreach g,$(GUIDES),$(REPO)/guide/$(g).pdf $(REPO)/guide/$(g).html)
