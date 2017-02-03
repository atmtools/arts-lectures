SUBDIRS = 01-rotational_spectra \
          02-absorption \
          03-line_shape \
          04-rtcalc \
          05-jacobian \
          06-olr

.PHONY: $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	@$(MAKE) -wC $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -wC $$dir clean; \
	done

cleanall:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -wC $$dir cleanall; \
	done
