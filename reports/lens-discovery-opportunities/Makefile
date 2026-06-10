# =============================================================================
# Makefile — build the Strong-Lens Discovery Opportunity report.
#
# Targets:
#   pdf       Build main.pdf via 4-pass pdflatex/bibtex.
#   figures   Regenerate the figures (footprint overlap map + priority chart).
#   clean     Remove LaTeX aux artefacts but keep main.pdf.
#   distclean Remove aux artefacts AND main.pdf.
#
# Same skeleton as reproductions/*/papers/Makefile; reuses the shared
# reproductions/tech-report.sty preamble two levels up.
# =============================================================================
PAPER       := main
TEX_SOURCE  := $(PAPER).tex
BIB_SOURCE  := references.bib
SHARED_STY  := ../../reproductions/tech-report.sty
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex
# Any matplotlib-enabled interpreter; the repo's venvs all carry mpl 3.10 / numpy 2.4.
PY          := /home/benson/.venvs/lens/bin/python

.PHONY: pdf figures clean distclean help

pdf: $(PAPER).pdf

$(PAPER).pdf: $(TEX_SOURCE) $(BIB_SOURCE) $(SHARED_STY)
	$(PDFLATEX) $(PAPER)
	$(BIBTEX)   $(PAPER)
	$(PDFLATEX) $(PAPER)
	$(PDFLATEX) $(PAPER)

figures:
	$(PY) figures/make_figures.py

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).log $(PAPER).out $(PAPER).fdb_latexmk \
	      $(PAPER).fls $(PAPER).toc $(PAPER).synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

help:
	@echo "Targets:"
	@echo "  pdf        Build $(PAPER).pdf"
	@echo "  figures    Regenerate figures via figures/make_figures.py"
	@echo "  clean      Remove auxiliary build artifacts (keep $(PAPER).pdf)"
	@echo "  distclean  Remove auxiliary build artifacts AND $(PAPER).pdf"
