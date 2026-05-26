# -----------------------------------------------------------------------------
# Makefile — build the Foundry I reproduction paper.
#
# Default target builds main.pdf via latexmk if available; otherwise falls
# back to the classic pdflatex -> bibtex -> pdflatex -> pdflatex 4-pass.
#
# Targets:
#   pdf       Build main.pdf (requires aastex701.cls from a recent TeX Live).
#   clean     Remove auxiliary build artifacts (keep main.pdf).
#   distclean Remove auxiliary build artifacts AND main.pdf.
# -----------------------------------------------------------------------------

# Force TeX Live 2026 (which ships aastex701.cls) ahead of any older system
# install at /usr/bin/pdflatex (e.g., TL2020 on AlmaLinux 9.7).
TL2026_BIN  := /usr/local/texlive/2026/bin/aarch64-linux
export PATH := $(TL2026_BIN):$(PATH)

PAPER       := main
TEX_SOURCE  := $(PAPER).tex
BIB_SOURCE  := references.bib
LATEXMK     := $(shell command -v latexmk 2> /dev/null)
PDFLATEX    := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX      := bibtex

.PHONY: pdf clean distclean help

pdf: $(PAPER).pdf

$(PAPER).pdf: $(TEX_SOURCE) $(BIB_SOURCE)
ifeq ($(LATEXMK),)
	@echo "latexmk not found — falling back to pdflatex/bibtex 4-pass."
	$(PDFLATEX) $(PAPER)
	$(BIBTEX)  $(PAPER)
	$(PDFLATEX) $(PAPER)
	$(PDFLATEX) $(PAPER)
else
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex
endif

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).fdb_latexmk $(PAPER).fls $(PAPER).synctex.gz \
	      $(PAPER).toc $(PAPER).lof $(PAPER).lot $(PAPER).run.xml

distclean: clean
	rm -f $(PAPER).pdf

help:
	@echo "Targets:"
	@echo "  pdf        Build $(PAPER).pdf"
	@echo "  clean      Remove auxiliary build artifacts (keep $(PAPER).pdf)"
	@echo "  distclean  Remove auxiliary build artifacts AND $(PAPER).pdf"
