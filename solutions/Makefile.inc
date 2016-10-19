# Variables
TC = latexmk
TFLAGS = -e "$$pdflatex=q/pdflatex -interaction=nonstopmode/" -pdf

# Targets
.PHONY: pdf

pdf:
	@$(TC) $(TFLAGS)

clean:
	$(TC) -silent -c

cleanall:
	$(TC) -silent -C

# vim:ft=make
#
