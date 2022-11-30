TARGETS = 01-absorption.pdf 02-lineshape.pdf 04-rtcalc.pdf 05-jacobian.pdf 06-oem.pdf 07-olr.pdf 08-scattering.pdf 09-heating_rates.pdf
CONVERT = jupyter nbconvert --log-level=ERROR --to pdf --output-dir=. --output=$@ --execute

check: convert

convert: ${TARGETS}

01-absorption.pdf: ../exercises/01-molecule_spectra/absorption.ipynb
	${CONVERT} $<

02-lineshape.pdf: ../exercises/02-line_shape/lineshape.ipynb
	${CONVERT} $<

04-rtcalc.pdf: ../exercises/04-rtcalc/rtcalc.ipynb
	${CONVERT} $<

05-jacobian.pdf: ../exercises/05-jacobian/jacobian.ipynb
	${CONVERT} $<

06-oem.pdf: ../exercises/06-inversion/oem.ipynb
	${CONVERT} $<

07-olr.pdf: ../exercises/07-olr/olr.ipynb
	${CONVERT} $<

08-scattering.pdf: ../exercises/08-scattering/scattering.ipynb
	${CONVERT} $<

09-heating_rates.pdf: ../exercises/09_heating_rates/heating_rates.ipynb
	${CONVERT} $<

clean:
	rm -f ${TARGETS}
