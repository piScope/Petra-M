PYTHON ?= $(shell which python)

INSTALLDIR=$(shell echo $(PetraM))
ifeq ($(INSTALLDIR),)
   INSTALLDIR := /usr/local/PetraM
endif
BINDIR=$(INSTALLDIR)/bin
LIBDIR=$(INSTALLDIR)/lib

default: compile

compile:
	$(PYTHON) setup.py build
install:
	$(PYTHON) setup.py install --prefix=$(INSTALLDIR)
clean:
	$(PYTHON) setup.py clean
