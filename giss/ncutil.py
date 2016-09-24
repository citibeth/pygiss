# PyGISS: Misc. Python library
# Copyright (c) 2013-2016 by Elizabeth Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import netCDF4
import sys

# Copy a netCDF file (so we can add more stuff to it)
class copy_nc(object):
    def __init__(self, nc0, ncout,
        var_filter=lambda x : x,
        attrib_filter = lambda x : True):
        """var_filter : function(var_name) -> new_var_name
            Only copy variables where this filter returns True.
        attrib_filter : function(attrib_name) -> bool
            Only copy attributes where this filter returns True."""
        self.nc0 = nc0
        self.ncout = ncout
        self.var_filter = var_filter
        self.attrib_filter = attrib_filter
        self.avoid_vars = set()
        self.avoid_dims = set()

    def createDimension(self, dim_name, *args, **kwargs):
        self.avoid_dims.add(dim_name)
        return self.ncout.createDimension(dim_name, *args, **kwargs)

    def copyDimensions(self, *dim_names):
        for dim_name in dim_names:
            l = len(self.nc0.dimensions[dim_name])
            self.ncout.createDimension(dim_name, l)

    def createVariable(self, var_name, *args, **kwargs):
        self.avoid_vars.add(var_name)
        return self.ncout.createVariable(var_name, *args, **kwargs)

    def define_vars(self, **kwargs):
        self.vars = self.nc0.variables.keys()

        # Figure out which dimensions to copy
        copy_dims = set()
        for var in self.vars:
            if not self.var_filter(var) : continue
            if var in self.avoid_vars : continue
            for dim in self.nc0.variables[var].dimensions:
                copy_dims.add(dim)

        # Copy the dimensions!
        for dim_pair in self.nc0.dimensions.items():
            name = dim_pair[0]
            extent = len(dim_pair[1])
            if name in copy_dims and name not in self.ncout.dimensions:
                self.ncout.createDimension(name, extent)

        # Define the variables
        for var_name in self.vars:
            ovname = self.var_filter(var_name)
            if ovname is None: continue
            if var_name in self.avoid_vars : continue

            var = self.nc0.variables[var_name]
            varout = self.ncout.createVariable(ovname, var.dtype, var.dimensions, **kwargs)
            for aname, aval in var.__dict__.items():
                if not self.attrib_filter(aname) : continue
                varout.setncattr(aname, aval)

    def copy_data(self):
        # Copy the variables
        for var_name in self.vars:
            ovname = self.var_filter(var_name)
            if ovname is None: continue
            if var_name in self.avoid_vars: continue

            ivar = self.nc0.variables[var_name]
            ovar = self.ncout.variables[ovname]
            ovar[:] = ivar[:]

def default_diff_fn(var, val0, val1):
    """Called when we see a difference"""
    pass

def diff(nc0, nc1, ncout=None,
        var_filter=lambda x : x,
        rtol=None, atol=None, equal_nan=None,    # np.isclose()
         **kwargs):    # nc.createVariable()
    """Finds differences between two NetCDF files.
    Optional args: rtol, atol, equal_nan.  See nump.isclose()"""

    isclose_kwargs = dict()
    if rtol is not None:
        isclose_kwargs['rtol'] = rtol
    if atol is not None:
        isclose_kwargs['atol'] = atol
    if equal_nan is not None:
        isclose_kwargs['equal_nan'] = equal_nan


    opened = list()

    try:
        if not isinstance(nc0, netCDF4.Dataset):
            nc0 = netCDF4.Dataset(nc0, 'r')
            opened.append(nc0)

        if not isinstance(nc1, netCDF4.Dataset):
            nc1 = netCDF4.Dataset(nc1, 'r')
            opened.append(nc1)

        if ncout is not None:
            if not isinstance(ncout, netCDF4.Dataset):
                ncout = netCDF4.Dataset(ncout, 'w', clobber=True)
                opened.append(ncout)

        extra0 = list()
        extra1 = list()
        diffs = list()

        remain1 = set([var for var in nc1.variables if var_filter(var) is not None])
        vars0 = [var for var in nc0.variables if var_filter(var) is not None]

        for var in vars0:
            if var not in remain1:
                extra0.append(var)
            else:
                val0 = nc0.variables[var][:]
                val1 = nc1.variables[var][:]
                if not np.isclose(val0, val1, **isclose_kwargs).all():
                    diffs.append(var)

            remain1.remove(var)

        for var in remain1:
            extra1.append(var)

        # Write out if we're given an output file
        diffs_set = set(diffs)
        if ncout is not None:
            nccopy = copy_nc(nc0, ncout,
                var_filter=lambda x : x if x in diffs_set else None)
            nccopy.define_vars(**kwargs)

            for var in diffs:
                val0 = nc0.variables[var][:]
                val1 = nc1.variables[var][:]
                ncout[var][:] = val1 - val0


        return (extra0, extra1, diffs)

    finally:
        for nc in opened:
            try:
                nc.close()
            except Exception as e:
                sys.stderr.write('Exception in ncutil.py diff(): {}\n'.format(e))
