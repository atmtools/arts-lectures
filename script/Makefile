# Variables
TC = latexmk
TFLAGS = -e "$$pdflatex=q/pdflatex -interaction=nonstopmode/" -pdf -bibtex

TARGETS = AdvRaRe_script_print.pdf AdvRaRe_script_screen.pdf

.PHONY: all $(TARGETS)

all: $(TARGETS)

$(TARGETS): %.pdf : %.tex
	$(TC) $(TFLAGS) $<

clean:
	$(TC) -silent -c

cleanall:
	$(TC) -silent -C

# vim:ft=make
#
