
all : snowdrift.so

-include f90_deps.mk

deps : f90_deps.mk
.PHONY : all deps

f90_deps.mk : $(SNOWDRIFT_LIB_SRC_F90) $(wildcard smain/*.f90 smain/*.F90)
	f90_mod_deps.py $(SNOWDRIFT_LIB_SRC_F90) $(wildcard smain/*.f90 smain/*.F90) >f90_deps.mk

# ==============================================
# Third Party Library Dependencies

# Must be Fortran-enabled
NETCDF_HOME=$(HOME)/opt/netcdf-4.1.3
BLITZ_HOME=$(HOME)/opt/blitz
CGAL_HOME=/opt/local
BOOST_HOME=/opt/local
GALAHAD=$(HOME)/qp/galahad/galahad
GALAHAD_ARCH=mac.osx.gfo
PYTHON_HOME=/opt/local/Library/Frameworks/Python.framework/Versions/2.7
PYTHON_INCLUDE=$(PYTHON_HOME)/include/python2.7
NUMPY_INCLUDE = $(PYTHON_HOME)/lib/python2.7/site-packages/numpy/core/include/numpy

# ==============================================
# Compilation Rules

FORTRAN_COMMAND_LINE = gfortran $(FFLAGS) -g -c -o $(basename $<).o
FORTRAN_INCLUDES = -I$(GALAHAD)/modules/$(GALAHAD_ARCH)/double -I$(NETCDF_HOME)/include -I.
FFLAGS = -shared -fPIC $(FORTRAN_INCLUDES)

CXX_INCLUDES = -Iinclude -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I$(BLITZ_HOME)/include
CXXFLAGS = -shared -fPIC -MD -g -frounding-math --std=c++0x $(CXX_INCLUDES)
C_INCLUDES = -Iinclude -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)
CFLAGS = -shared -fPIC -MD -g -frounding-math $(C_INCLUDES)

CC = gcc

#%.o : %.F90
#	gfortran $(FFLAGS) -c $<
#%.o : %.f90
#	gfortran $(FFLAGS) -c $<

# ================= The library

SNOWDRIFT_LIB_SRC_F90 = $(wildcard slib/*.f90 slib/*.F90)
SNOWDRIFT_LIB_SRC = $(wildcard slib/*.cpp slib/*.c) $(SNOWDRIFT_LIB_SRC_F90)
SNOWDRIFT_LIB_OBJ = $(addsuffix .o,$(basename $(SNOWDRIFT_LIB_SRC)))
SNOWDRIFT_LIB_LIBS = \
	-L$(GALAHAD)/objects/$(GALAHAD_ARCH)/double -lgalahad \
	-lgalahad_hsl -lgalahad_pardiso -lgalahad_wsmp -lgalahad_metis \
	-lgalahad_lapack -lgalahad_blas -lgomp \
	-L$(NETCDF_HOME)/lib -lnetcdff -lnetcdf -lnetcdf_c++ \
	-L$(CGAL_HOME)/lib -lCGAL -lmpfr -lgmp -lproj \
	-L$(BOOST_HOME)/lib -lboost_thread-mt \
	-L$(BLITZ_HOME)/lib -lblitz \
	-lpython2.7


test :
	echo $(SNOWDRIFT_LIB_SRC_F90)

# Pull in dependency info for existing .o files
-include $(SNOWDRIFT_LIB_OBJ:.o=.d)

snowdrift.so : $(SNOWDRIFT_LIB_OBJ)
	gfortran -shared -Wl -o snowdrift.so $(SNOWDRIFT_LIB_OBJ) $(SNOWDRIFT_LIB_LIBS) -lstdc++

searise_ll_overlap : smain/searise_ll_overlap.o snowdrift.so
	g++ -o $@ $^ $(SNOWDRIFT_LIB_LIBS)

xy_overlap : smain/xy_overlap.o snowdrift.so
	g++ -o $@ $^ $(SNOWDRIFT_LIB_LIBS)

xy_snowdrift : smain/xy_snowdrift.o snowdrift.so
	gfortran -o $@ $^ $(SNOWDRIFT_LIB_LIBS) -lstdc++

hctest : smain/hctest.o snowdrift.so
	g++ -o $@ $^ $(SNOWDRIFT_LIB_LIBS)

pytypes : smain/pytypes.o snowdrift.so
	g++ -o $@ $^ $(SNOWDRIFT_LIB_LIBS)

read_eqp : smain/read_eqp.o slib/ncutil_f.o
	gfortran -o $@ $^ \
	-L$(NETCDF_HOME)/lib -lnetcdff -lnetcdf \
	-L$(GALAHAD)/objects/$(GALAHAD_ARCH)/double -lgalahad \
	-lgalahad_hsl -lgalahad_pardiso -lgalahad_wsmp -lgalahad_metis \
	-lgalahad_lapack -lgalahad_blas -lgomp


clean :
	-rm `find . -name '*.d'`
	-rm `find . -name '*.o'`
	-rm `find . -name '*.mod'`
	-rm `find . -name '*.so'`

% =================================================

fig/%.png : fig/%.py
	python $^

figures : fig/gcm_grid.png fig/gcm_grid_hc.png fig/elevation.png
