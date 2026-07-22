# =============================================================================
# Makefile — build the claude-giga-lens-linus technical report.
#
#   main.pdf   — Bridging a Correlated-Noise Likelihood and Evidence-Based
#                Inference onto the Next-Generation GIGA-Lens Scene API.
#
# Single document sharing the repo-wide preamble ../../tech-report.sty
# (natbib + bibtex, author-year; see that file for why not biblatex+biber on
# this host). Builds with vanilla TeX Live (>= 2022); no special class file.
# Same skeleton as the claude-giga-lens papers/Makefile.
# =============================================================================
PAPERS      := main
SECTIONS    := sec_a_front_methods.tex sec_summary_modelers.tex sec_b_results_mech.tex sec_c_bench_disc.tex sec_appendix_convergence.tex
BIB_SOURCE  := references.bib
SHARED_STY  := ../../tech-report.sty
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf clean distclean help

pdf: $(addsuffix .pdf,$(PAPERS))

# Four pdflatex passes: this document's float pages shift between passes 2
# and 3, so a third post-bibtex pass is needed for stable cross-references.
%.pdf: %.tex $(SECTIONS) $(BIB_SOURCE) $(SHARED_STY)
	$(PDFLATEX) $*
	$(BIBTEX)   $*
	$(PDFLATEX) $*
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
