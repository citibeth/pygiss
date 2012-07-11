# this is example-file: src/bin/Makefile.am

include ../commonmk.am

###############################################################################
# THE PROGRAMS TO BUILD
###############################################################################

# the program to build (the names of the final binaries)
bin_PROGRAMS = searise_ll_overlap

# list of sources for the 'helloWorld' binary
searise_ll_overlap_SOURCES = searise_ll_overlap.cpp

# the additional libraries needed to link helloWorld
searise_ll_overlap_LDADD = 	$(top_builddir)/slib/libsnowdrift.la $(AM_LDFLAGS)

###############################################################################
