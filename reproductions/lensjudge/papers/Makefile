# =============================================================================
# Makefile — build the LensJudge tech reports (4-pass pdflatex/bibtex).
#   main     : the main LensJudge report (main.tex)
#   residual : the residual-image transformation report (residual.tex)
# Same skeleton as reproductions/claudenet/papers/Makefile (multi-target).
# =============================================================================
PAPER       := main
RESIDUAL    := residual
BIB_SOURCE  := references.bib
SHARED_STY  := ../../tech-report.sty
# The PDFs \includegraphics the figures/*.png (v1/v2/v3 figures from eval/make_*_figures.py
# and the residual figures from eval/make_residual_figures.py) and the loose euclid_flip.png,
# so regenerating a figure correctly triggers a rebuild.
FIG_DEPS    := $(wildcard figures/*.png) euclid_flip.png
DEPS        := $(BIB_SOURCE) $(SHARED_STY) $(FIG_DEPS)
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf residual all clean distclean help

pdf: $(PAPER).pdf
residual: $(RESIDUAL).pdf
all: pdf residual

$(PAPER).pdf: $(PAPER).tex $(DEPS)
	$(PDFLATEX) $(PAPER)
	$(BIBTEX)   $(PAPER)
	$(PDFLATEX) $(PAPER)
	$(PDFLATEX) $(PAPER)

$(RESIDUAL).pdf: $(RESIDUAL).tex $(DEPS)
	$(PDFLATEX) $(RESIDUAL)
	$(BIBTEX)   $(RESIDUAL)
	$(PDFLATEX) $(RESIDUAL)
	$(PDFLATEX) $(RESIDUAL)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.toc *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf $(RESIDUAL).pdf

help:
	@echo "Targets: pdf (main) | residual | all | clean | distclean"
