# =============================================================================
# Makefile -- build the AION-1 reproduction tech report.
# Same skeleton as the sibling reproductions (foundry-i, hsu-2025, ...).
# Builds with vanilla TeX Live (>= 2022); no special class file or PATH pin.
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
	@echo "Targets: pdf | clean | distclean"
