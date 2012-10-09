import odict
import numpy as np
import netCDF4
import giss.ncutil

# -----------------------------------------------------------------

# -------------------------------------------------------------
# -------------------------------------------------------------
# Reads info about a variable, no matther whether that variable is
# sitting in a netCDF file, or was read in from a GISS-format file
# @return (val, sdims, dtype)
#   val = The np.array values of the variable
#   sdim = Tuple of names of the variable's dimensions
#   dtype = Type of the variable
def gread(handle, var_name) :
	if isinstance(handle, odict.odict) :	# Just fetch the variable from topo
		return handle[var_name]
	else :		# We have a netcdf handle
		return giss.ncutil.read_ncvar_struct(handle, var_name)

# Reads info about a variable, no matther whether that variable is
# sitting in a netCDF file, or was read in from a GISS-format file
# @return (sdims, shape, dtype)
#   shape = Dimensions of the np.array that would hold the variable
#   sdim = Tuple of names of the variable's dimensions
#   dtype = Type of the variable
def gread_dims(handle, name) :
#	print 'gread_dims',name,str(handle.__class__)
	if isinstance(handle, odict.odict) :	# Just fetch the variable from topo
		tuple = handle[name]		# (value, sdims) pairs
		sdims = tuple.sdims
		shape = tuple.val.shape
		dtype = tuple.dtype
	elif isinstance(handle, netCDF4.Dataset) :
		print 'Fetching netCDF variable = "%s"' % name
		ncvar = handle.variables[name]
		sdims = ncvar.dimensions
		shape = ncvar.shape
		dtype = ncvar.dtype
	else :		# We have a netcdf handle
		raise Exception('Unsupported data type for handle of class %s' % str(handle.__class__))
	return (sdims, shape, dtype)
# -------------------------------------------------------------
# Given a bunch of variables that have been determined, writes
# them to a netCDF file.  Defines the dimensions, etc. as needed
# @param ofname Name of netCDF file to write
# @param wvars = list of variables to write
#        wvars[i] = (handle, name)
#        handle = Way to look up the variable (TOPO array, netCDF handle, etc)
#        name = Name of variable
def write_netcdf(ofname, wvars) :
	print 'Writing netCDF File: %s' % ofname
	nc = netCDF4.Dataset(ofname, 'w', format='NETCDF3_CLASSIC')

	# Collect together the dimension names and lengths for each variable
	wdims = odict.odict()
	for vv in wvars :		# (handle, name) pairs
		handle = vv[0]
		name = vv[1]
		sdims, shape, dtype = gread_dims(handle, name)

		for i in range(0, len(sdims)) :
			sdim = sdims[i]
			shp = shape[i]
			if sdim in wdims :
				if shp != wdims[sdim] :
					raise Exception('Inconsistent dimension for %s (%d vs %d)' % (sdim, shp, wdims[sdim]))
			else :
				wdims[sdim] = shp
	print wdims

	# Define the dimensions
	for name, val in wdims.iteritems() :
		print name, val
		nc.createDimension(name, val)

	# Define the variables
	for vv in wvars :		# (handle, name) pairs
		handle = vv[0]
		name = vv[1]
		sdims, shape, dtype = gread_dims(handle, name)
		nc.createVariable(name, dtype, sdims)

	# Copy in the data
	for vv in wvars :		# (handle, name) pairs
		handle = vv[0]
		name = vv[1]
		tuple = gread(handle, name)
		nc.variables[name][:] = tuple.val[:]

	nc.close()
