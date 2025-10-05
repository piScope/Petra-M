SUBDIRS = solver/mumps

default:
	for dirs in $(SUBDIRS); do\
		$(MAKE) -C $$dirs;\
	done
all:
	for dirs in $(SUBDIRS); do\
		$(MAKE) -C $$dirs all;\
	done
cxx:
	for dirs in $(SUBDIRS); do\
		$(MAKE) -C $$dirs cxx;\
	done
cleancxx:
	for dirs in $(SUBDIRS); do\
		$(MAKE) -C $$dirs cleancxx;\
	done
clean:
	for dirs in $(SUBDIRS); do\
		$(MAKE) -C $$dirs clean;\
	done
