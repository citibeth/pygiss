# this is example-file: src/snowdrift/Makefile.am

include ../commonmk.am

###############################################################################
# THE LIBRARIES TO BUILD
###############################################################################

# the library names to build (note we are building static libs only)
lib_LTLIBRARIES = libsnowdrift.la

# the sources to add to the library and to add to the source distribution
libsnowdrift_la_SOURCES = \
	Grid.cpp GridCell.cpp Grid_LatLon.cpp Grid_XY.cpp HeightClassifier.cpp IndexTranslator.cpp OverlapMatrix.cpp Rasterizer.cpp Snowdrift.cpp SparseMatrix.cpp clippers.cpp geodesy.cpp hsl_zd11_x.cpp maputils.cpp ncutil.cpp qpt_x.cpp

# ------------------------------------------------------
if HAVE_GALAHAD
-include f90_deps.mk
FORTRAN_COMMAND_LINE = $(FCCOMPILE) -c -o

deps : f90_deps.mk
.PHONY : all deps

f90_deps.mk : ../f90_mod_deps.py $(FORTRAN_SOURCES)
	python ../f90_mod_deps.py $(FORTRAN_SOURCES) >f90_deps.mk

FORTRAN_SOURCES = eqp_f.f90 hsl_zd11d.f90 hsl_zd11d_f.f90 ncutil_f.f90 qpt_f.f90 x_loc_f.f90
libsnowdrift_la_SOURCES += $(FORTRAN_SOURCES)
endif
# ------------------------------------------------------



# PY_SOURCES = pyutil.cpp Grid_py.cpp Rasterizer_py.cpp Snowdrift_py.cpp snowdrift_module_py.cpp


# libsnowdrift_la_SOURCES = $(shell find -E . -regex '.*\.(cpp|c|f90|F90|f|F)')
						
###############################################################################
