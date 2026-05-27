# =============================================================================
# Makefile — build Phase 3a (Huang 2020) internal reproduction report.
#
# Targets:
#   pdf       Build main.pdf via latexmk if available, else 4-pass fallback.
#   clean     Remove LaTeX aux artefacts but keep main.pdf.
#   distclean Remove aux artefacts AND main.pdf.
#
# Requires:
#   pdflatex (TeX Live >= 2022)
#   biber    (for biblatex bibliography)
#   article  (standard LaTeX class, ships with every TeX Live)
#
# No special class file or TL version pin — vanilla TeX Live works.
# =============================================================================
PAPER       := main
TEX_SOURCE  := $(PAPER).tex
BIB_SOURCE  := references.bib
SHARED_STY  := ../../tech-report.sty
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf clean distclean help

pdf: $(PAPER).pdf

$(PAPER).pdf: $(TEX_SOURCE) $(BIB_SOURCE) $(SHARED_STY)
	$(PDFLATEX) $(PAPER)
	$(BIBTEX)   $(PAPER)
	$(PDFLATEX) $(PAPER)
	$(PDFLATEX) $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).log $(PAPER).out $(PAPER).fdb_latexmk \
	      $(PAPER).fls $(PAPER).toc $(PAPER).synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

help:
	@echo "Targets:"
	@echo "  pdf        Build $(PAPER).pdf"
	@echo "  clean      Remove auxiliary build artifacts (keep $(PAPER).pdf)"
	@echo "  distclean  Remove auxiliary build artifacts AND $(PAPER).pdf"
