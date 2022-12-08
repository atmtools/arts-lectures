TARGETS = 01-absorption.pdf \
		  02-lineshape.pdf \
		  04-rtcalc.pdf \
		  05-jacobian.pdf \
		  06-oem.pdf \
		  07-olr.pdf \
		  08-scattering.pdf \
		  09-heating_rates.pdf

SOURCES = ../exercises/01-molecule_spectra/absorption.ipynb \
		  ../exercises/02-line_shape/lineshape.ipynb \
		  ../exercises/04-rtcalc/rtcalc.ipynb \
		  ../exercises/05-jacobian/jacobian.ipynb \
		  ../exercises/06-inversion/oem.ipynb \
		  ../exercises/07-olr/olr.ipynb \
		  ../exercises/08-scattering/scattering.ipynb \
		  ../exercises/09_heating_rates/heating_rates.ipynb

CONVERT = jupyter nbconvert --log-level=ERROR --to pdf --output-dir=. --output=$@ --execute


all: convert

define make-exercise-target
  convert:: $1
  $1: $2
endef

ILIST = $(shell for x in {1..$(words $(TARGETS))}; do echo $$x; done)
$(foreach i,$(ILIST), \
	$(eval $(call make-exercise-target, \
		$(word $(i),$(TARGETS)), \
		$(word $(i),$(SOURCES)))))

$(TARGETS):
	TEXINPUTS=$(PWD)/$(dir $<): $(CONVERT) $<

clean:
	rm -f $(TARGETS)
