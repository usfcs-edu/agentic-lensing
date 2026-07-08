# =============================================================================
# Makefile — build the claude-giga-lens technical report.
#
#   main.pdf   — Correlated-noise likelihoods and next-generation inference
#                for strong gravitational lens modeling.
#
# Single document sharing the repo-wide preamble ../../tech-report.sty
# (natbib + bibtex, author-year; see that file for why not biblatex+biber on
# this host). Builds with vanilla TeX Live (>= 2022); no special class file.
# Same skeleton as the foundry-i papers/Makefile.
# =============================================================================
PAPERS      := main
BIB_SOURCE  := references.bib
SHARED_STY  := ../../tech-report.sty
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf clean distclean help

pdf: $(addsuffix .pdf,$(PAPERS))

%.pdf: %.tex $(BIB_SOURCE) $(SHARED_STY)
	$(PDFLATEX) $*
	$(BIBTEX)   $*
	$(PDFLATEX) $*
	$(PDFLATEX) $*

clean:
	rm -f $(foreach p,$(PAPERS),\
	      $(p).aux $(p).bbl $(p).blg $(p).log $(p).out \
	      $(p).fdb_latexmk $(p).fls $(p).toc $(p).synctex.gz)

distclean: clean
	rm -f $(addsuffix .pdf,$(PAPERS))

help:
	@echo "Targets:"
	@echo "  pdf        Build $(addsuffix .pdf,$(PAPERS))"
	@echo "  clean      Remove auxiliary build artifacts (keep PDFs)"
	@echo "  distclean  Remove auxiliary build artifacts AND PDFs"
