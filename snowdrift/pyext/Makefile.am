
include ../commonmk.am

# PYTHON_PREFIX := $(shell $(PYTHON)-config --prefix)

PYTHON_CFLAGS = $(shell $(PYTHON)-config --cflags)
NUMPY_CFLAGS = $(shell $(PYTHON) -c "import numpy; print '-I' + numpy.get_include()")

AM_CXXFLAGS += $(PYTHON_CFLAGS) $(NUMPY_CFLAGS)
pyexec_LTLIBRARIES = snowdrift.la
snowdrift_la_SOURCES = Grid_py.cpp Rasterizer_py.cpp Snowdrift_py.cpp snowdrift_module_py.cpp
snowdrift_la_LDFLAGS = -module $(PYTHON_LDFLAGS) $(NUMPY_LDFLAGS) $(AM_LDFLAGS) $(top_srcdir)/slib/libsnowdrift.la

