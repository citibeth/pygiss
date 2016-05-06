# pyGISS: GISS Python Library
# Copyright (c) 2013 by Robert Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import odict
import numpy as np
import netCDF4
import giss.ncutil

"""Basic I/O routines to support the height-classification of GIC and
TOPO files for ModelE.

These might be more general than just that purpose.  But then again,
they might not.  For now, consider this module to be internal."""

# -----------------------------------------------------------------
# -----------------------------------------------------------------
def _check_shape(var, dims, varname) :
    """Check dimensions of a numpy array

    Args:
        var (np.array):
            Variable to check
        dims (tuple of int):
            Expected shape of the variable

    Raises:
        Exception, if var.shape != dims"""
    if var.shape != dims :
        raise Exception('%s%s should have dimensions %s' % (varname, str(dims), str(var.shape)))


# -------------------------------------------------------------
def _read_ncvar_struct(nc, var_name, output_dtype=float) :
    """Read a variable from a netCDF file, along with metadata
    Args:
        nc (netCDF4.Dataset):
            Handle to open netCDF file toread
        var_name (string):
            Name of variable to read
        output__dtype (dtype):
            
    Returns:
        .name (string):
            Name of the variable in the netCDF file
        .val (np.array, dtype=output_dtype):
            Value of the variable (netCDF)
        .sdims[] (string):
            Names of dimensions of .val
        .dtype:
            Data type of the variable in the netCDF file"""

    ncvar = nc.variables[var_name]

    if output_dtype is None :
        output_dtype = ncvar.dtype

    if ncvar.shape == () :  # Scalar variable
        val = ncvar.getValue()
    else :          # Array variable
        val = np.zeros(ncvar.shape, dtype=output_dtype)
        val[:] = ncvar[:]
    out = {'name' : var_name, 'val' : val, 'sdims' : ncvar.dimensions, 'dtype' : ncvar.dtype}
    return giss.util.Struct(out)

# -------------------------------------------------------------
# Reads all tuples from a GISS-format file (the TOPO file)
# @return A odict.odict() topo[name] = {.name, .val, .sdims, .dtype}
def _read_all_giss_struct(fname) :
    topo = odict.odict()
    for rec in giss.gissfile.reader(fname) :
        val = np.zeros(rec.data.shape)  # Promote to double
        name = rec.var.lower()
        val[:] = rec.data[:]
        topo[name] = giss.util.Struct({
            'name' : name,
            'val' : val,
            'sdims' : (u'jm', u'im'),
            'dtype' : 'f8'})
    return topo
# -------------------------------------------------------------
# Reads info about a variable, no matther whether that variable is
# sitting in a netCDF file, or was read in from a GISS-format file
# @return (val, sdims, dtype)
#   val = The np.array values of the variable
#   sdim = Tuple of names of the variable's dimensions
#   dtype = Type of the variable
def gread(handle, var_name) :
    if isinstance(handle, odict.odict) :    # Just fetch the variable from topo
        return handle[var_name]
    else :      # We have a netcdf handle
        return _read_ncvar_struct(handle, var_name, float)

# Reads info about a variable, no matther whether that variable is
# sitting in a netCDF file, or was read in from a GISS-format file
# @return (sdims, shape, dtype)
#   shape = Dimensions of the np.array that would hold the variable
#   sdim = Tuple of names of the variable's dimensions
#   dtype = Type of the variable
def gread_dims(handle, name) :
#   print 'gread_dims',name,str(handle.__class__)
    if isinstance(handle, odict.odict) :    # Just fetch the variable from topo
        tuple = handle[name]        # (value, sdims) pairs
        sdims = tuple.sdims
        shape = tuple.val.shape
        dtype = tuple.dtype
    elif isinstance(handle, netCDF4.Dataset) :
        print 'Fetching netCDF variable = "%s"' % name
        ncvar = handle.variables[name]
        sdims = ncvar.dimensions
        shape = ncvar.shape
        dtype = ncvar.dtype
    else :      # We have a netcdf handle
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
    for vv in wvars :       # (handle, name) pairs
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
    for vv in wvars :       # (handle, name) pairs
        handle = vv[0]
        name = vv[1]
        sdims, shape, dtype = gread_dims(handle, name)
        nc.createVariable(name, dtype, sdims)

    # Copy in the data
    for vv in wvars :       # (handle, name) pairs
        handle = vv[0]
        name = vv[1]
        tuple = gread(handle, name)
        nc.variables[name][:] = tuple.val[:]

    nc.close()
