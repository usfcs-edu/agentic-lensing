# =============================================================================
# Makefile — build the LensJudge tech report (4-pass pdflatex/bibtex).
# Same skeleton as reproductions/hsu-2025/papers/Makefile.
# =============================================================================
PAPER       := main
TEX_SOURCE  := $(PAPER).tex
BIB_SOURCE  := references.bib
SHARED_STY  := ../../tech-report.sty
# The PDF \includegraphics the figures/*.png (v1 appendix + the 3 v2 figures from
# eval/make_v2_figures.py) and the loose euclid_flip.png, so regenerating a figure
# correctly triggers a rebuild.
FIG_DEPS    := $(wildcard figures/*.png) euclid_flip.png
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf clean distclean help

pdf: $(PAPER).pdf

$(PAPER).pdf: $(TEX_SOURCE) $(BIB_SOURCE) $(SHARED_STY) $(FIG_DEPS)
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
