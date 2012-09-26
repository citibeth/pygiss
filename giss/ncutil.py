import netCDF4
import numpy as np
import giss.util

# General utilitis for reading/writing netCDF files

# -----------------------------------------------------------------
# Efficiently reads an array from a netCDF file
# @param dtype Data type of output array
def read_ncvar(nc, varname, dtype=float) :
	ncvar = nc.variables[varname]
	ret = np.zeros(ncvar.shape,dtype=dtype)
	ret[:] = ncvar[:]
	return ret


# -----------------------------------------------------------------
# Read tuple from a netCDF file
# @return (val, sdims, dtype)
#   val = The np.array values of the variable
#   sdim = Tuple of names of the variable's dimensions
#   dtype = Type of the variable in netCDF
def read_ncvar_struct(nc, var_name, output_dtype=float) :
	ncvar = nc.variables[var_name]
	val = np.zeros(ncvar.shape, dtype=output_dtype)
	val[:] = ncvar[:]
	out = {'name' : var_name, 'val' : val, 'sdims' : ncvar.dimensions, 'dtype' : ncvar.dtype}
	return giss.util.Struct(out)
