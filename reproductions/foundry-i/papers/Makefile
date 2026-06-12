# =============================================================================
# Makefile — build the Foundry I reproduction reports.
#
# Two documents share references.bib and ../../tech-report.sty:
#   main.pdf       — concise final-state report (served on the project site)
#   evolution.pdf  — complete development record (v1 -> v13)
#
# Same skeleton as Phase 2 (hsu-2025) and Phase 3a (huang-2020) papers/.
# Builds with vanilla TeX Live (>= 2022); no special class file or PATH pin.
# =============================================================================
PAPERS      := main evolution
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
