# =============================================================================
# Makefile — build the Cikota+2023 Einstein cross internal reproduction report.
#
# Targets:
#   pdf       Build main.pdf via 4-pass pdflatex/bibtex.
#   clean     Remove LaTeX aux artefacts but keep main.pdf.
#   distclean Remove aux artefacts AND main.pdf.
#
# Same skeleton as reproductions/hsu-2025/papers/Makefile.
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
