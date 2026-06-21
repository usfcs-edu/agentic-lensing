# Build the DR11 campaign report. Vanilla TeX Live (>= 2022). `make pdf`.
PAPER    := main
TEX_DEPS := $(wildcard *.tex) $(wildcard tables/*.tex)
FIG_DEPS := $(wildcard figures/*.png)
DEPS     := $(TEX_DEPS) $(FIG_DEPS) references.bib ../../tech-report.sty
PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
.PHONY: pdf clean
pdf: $(PAPER).pdf
$(PAPER).pdf: $(DEPS)
	$(PDFLATEX) $(PAPER)
	bibtex $(PAPER)
	$(PDFLATEX) $(PAPER)
	$(PDFLATEX) $(PAPER)
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.toc *.synctex.gz
