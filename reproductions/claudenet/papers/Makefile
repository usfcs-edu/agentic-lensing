# =============================================================================
# Makefile -- build the ClaudeNet research report.
# Same skeleton as the sibling reproductions (aion-1, foundry-i, hsu-2025, ...).
# Builds with vanilla TeX Live (>= 2022).
#
# NOTE: main.tex \input's modular section files (v2_section, campaign_section,
# v3_section, future_work, appendix_catalog) and \input's the auto-generated
# tables/*.tex, and \includegraphics the figures/*.png. The PDF therefore depends
# on ALL of those, not just main.tex -- so editing a section or regenerating a
# table/figure correctly triggers a rebuild (the previous Makefile tracked only
# main.tex and silently skipped section edits).
# =============================================================================
PAPER       := main
CORE        := v3_core
TEX_DEPS    := $(wildcard *.tex) $(wildcard tables/*.tex)
FIG_DEPS    := $(wildcard figures/*.png)
BIB_SOURCE  := references.bib
SHARED_STY  := ../../tech-report.sty
DEPS        := $(TEX_DEPS) $(FIG_DEPS) $(BIB_SOURCE) $(SHARED_STY)
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf core all clean distclean help

pdf: $(PAPER).pdf
core: $(CORE).pdf
all: pdf core

# 3x pdflatex + bibtex resolves \ref/\cite across the modular \input's.
$(PAPER).pdf: $(DEPS)
	$(PDFLATEX) $(PAPER)
	$(BIBTEX)   $(PAPER)
	$(PDFLATEX) $(PAPER)
	$(PDFLATEX) $(PAPER)

# Standalone liftable v3 paper-core (sibling-section \ref's resolve to `??').
$(CORE).pdf: $(DEPS)
	$(PDFLATEX) $(CORE)
	$(BIBTEX)   $(CORE)
	$(PDFLATEX) $(CORE)
	$(PDFLATEX) $(CORE)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.toc *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf $(CORE).pdf

help:
	@echo "Targets: pdf (main) | core (v3_core) | all | clean | distclean"
