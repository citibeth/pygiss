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
import os
import shutil
import functools,operator
from giss.functional import *
from giss import functional
from giss import checksum,memoize
import collections
import cf_units
import xarray

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

def install_nc(ifname, odir, installed=None):

    """Installs a netCDF file into odir.  Follows a convention for
    dependencies in the netCDF file:

        Attributes on the variable 'file' list files related to this
        file (absoulte paths).  Parallel attributes on the variable
        'install_paths' list the relative directory each file should
        be installed in.

    For example:

        int files ;
                files:source = "/home2/rpfische/modele_input/origin/GIC.144X90.DEC01.1.ext_1.nc" ;
                files:elev_mask = "/home2/rpfische/f15/modelE/init_cond/ice_sheet_ec/elev_mask.nc" ;
                files:icebin_in = "/home2/rpfische/f15/modelE/init_cond/ice_sheet_ec/icebin_in.nc" ;
        int install_paths ;
                install_paths:elev_mask = "landice" ;
                install_paths:icebin_in = "landice" ;

    Files without an install_path won't get installed."""

    if installed is None:
        installed = dict()    # ifname --> ofname

    # Make sure destination exists
    try:
        os.makedirs(odir)
    except:
        pass

    # Copy the netCDF file to the destination
    _,ifleaf = os.path.split(ifname)
    ofname = os.path.join(odir, ifleaf)
    print('Installing {} ->\n    {}'.format(ifname, ofname))
    shutil.copyfile(ifname, ofname)

    # Install dependencies of this file
    nc = None
    try:

        try:
            nc = netCDF4.Dataset(ofname, 'a')
        except RuntimeError:
            return

        try:
            files = nc.variables['files']
            install_paths = nc.variables['install_paths']
        except KeyError:
            # Files doesn't follow the conventions
            return

        # Iterate through attributes
        for label, relpath in install_paths.__dict__.items():
            child_ifname = getattr(files, label)
            _,child_leaf = os.path.split(child_ifname)
            child_odir = os.path.abspath(os.path.join(odir, relpath))
            child_ofname = os.path.join(child_odir, child_leaf)

            install_nc(child_ifname, child_odir)
            setattr(files, label, child_ofname)

    finally:
        if nc is not None:
            nc.close()
# -----------------------------------------------
# Stuff for xaccess

# -------------------------------------------------------------
@memoize.local()
def ncopen(name):
    nc = netCDF4.Dataset(name, 'r')
    return nc

# ---------------------------------------------------------------
def ncdata(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    """Simple accessor function for data in NetCDF files.
    Ops on this aren't very interesting because it is a
    fully-bound thunk."""

    nc = ncopen(fname)
    var = nc.variables[var_name]

    data = var[index]

    if not np.issubdtype(var.dtype, np.integer):
        # Missing value stuff only makes sense for floating point

        if missing_value is not None:
            # User override of NetCDF standard
            data[data == missing_value] = nan
        elif hasattr(var, 'missing_value'):
            # NetCDF standard
            data[data == var.missing_value] = nan
        elif missing_threshold is not None:
            # Last attempt to fix a broken file
            data[np.abs(data) > missing_threshold] = nan

    return data

# --------------------------------------------
def _fetch_shape(var, index):
    """Given a variable and indexing, determines the shape of the
    resulting fetch (without actually doing it)."""
    shape = []
    dims = []
    for i in range(0,len(var.shape)):
        if i >= len(index):    # Implied ':' for this dim
            dims.append(var.dimensions[i])
            shape.append(var.shape[i])
        else:
            ix = index[i]
            if isinstance(ix, slice):
                dims.append(var.dimensions[i])
                start = 0 if ix.start is None else ix.start
                stop = var.shape[i] if ix.stop is None else ix.stop
                step = 1 if ix.step is None else ix.step
                shape.append((stop - start) // step)
    return tuple(shape), tuple(dims)

FetchTuple = xnamedtuple('FetchTuple', ('attrs', 'data'))



@function()
def ncattrs(file_name, var_name):
    """Produces extended attributes on a variable fetch operation"""
    nc = ncopen(file_name)

    attrs = {}
    var = nc.variables[var_name]

    # User can retrieve nc.ncattrs(), etc.
    for key in nc.ncattrs():
        attrs[('file', key)] = getattr(nc, key)

    # User can retrieve var.dimensions, var.shape, var.name, var.xxx, var.ncattrs(), etc.

    attrs[('var', 'dimensions')] = var.dimensions
    attrs[('var', 'dtype')] = var.dtype
    attrs[('var', 'datatype')] = var.datatype
    attrs[('var', 'ndim')] = var.ndim
    attrs[('var', 'shape')] = var.shape
    attrs[('var', 'scale')] = var.scale
    # Don't know why this doesn't work.  See:
    # http://unidata.github.io/netcdf4-python/#netCDF4.Variable
    # attrs[('var', 'least_significant_digit')] = var.least_significant_digit
    attrs[('var', 'name')] = var.name
    attrs[('var', 'size')] = var.size
    for key in var.ncattrs():
        attrs[('var', key)] = getattr(var, key)


    return wrap_combine(attrs, intersect_dicts)

def add_fetch_attrs(attrs, file_name, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):

    nc = ncopen(file_name)
    var = nc.variables[var_name]

    attrs[('fetch', 'file_name')] = file_name
    attrs[('fetch', 'var_name')] = var_name
    attrs[('fetch', 'missing_value')] = missing_value
    attrs[('fetch', 'missing_threshold')] = missing_threshold
    attrs[('fetch', 'nan')] = np.nan
    attrs[('fetch', 'index')] = index
    fetch_shape,fetch_dims = _fetch_shape(var, index)
    attrs[('fetch', 'shape')] = fetch_shape
    attrs[('fetch', 'dimensions')] = fetch_dims
    attrs[('fetch', 'size')] = functools.reduce(operator.mul, fetch_shape, 1)
    attrs[('fetch', 'dtype')] = attrs[('var', 'dtype')]


# ----------------------------------------------------
# Filter: Return xarray instead of np.array
@functional.lift()
def data_to_xarray(attrs, data):
    """Converts the result of an ncfetch (attrs, data) into (attrs, xarray)"""
    dims = attrs[('fetch', 'dimensions')]
    return xarray.Variable(dims, data, attrs=attrs)

@function()
def ncfetch(file_name, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None, **kwargs):

    attrsW = ncattrs(file_name, var_name)
    add_fetch_attrs(attrsW(), file_name, var_name, **kwargs)

    return FetchTuple(
        attrsW,
            bind(ncdata, file_name, var_name, *index, **kwargs))
# ---------------------------------------------------------
@function()
def sum_fetch1(fetch, axis=None, dtype=None, out=None, keepdims=False):
    """Works like np.sum, but on a fetch record..."""
    attrs0 = fetch.attrs()
    data1 = fetch.data

    ashape = attrs0[('fetch', 'shape')]
    adims = attrs0[('fetch', 'dimensions')]

    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html
    # axis : None or int or tuple of ints, optional
    # 
    # Axis or axes along which a sum is performed. The default (axis =
    # None) is perform a sum over all the dimensions of the input
    # array. axis may be negative, in which case it counts from the
    # last to the first axis.
    #
    # If this is a tuple of ints, a sum is performed on multiple axes,
    # instead of a single axis or all the axes as before.
    if axis is None:
        axes = set(range(len(ashape)))
    if isinstance(axis, tuple):
        axes = set(axis)
    else:
        axes = set((axis,))

    shape = []
    dims = []
    for i in range(0,len(ashape)):
        if i in axes:
            if keepdims:
                shape.append(1)
                dims.append(adims[i])
        else:
            shape.append(ashape[i])
            dims.append(adims[i])

    attrs0[('fetch', 'shape')] = tuple(shape)
    attrs0[('fetch', 'dimensions')] = tuple(dims)

    data = functional.lift_once(np.sum, fetch.data, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    return FetchTuple(fetch.attrs, data)

sum_fetch2 = functional.lift()(sum_fetch1)
# ---------------------------------------------------------
_zero_one = np.array([0.,1.])

def convert_unitsV(fetch, ounits):
    """fetch:
        Result of the fetch() function

    V = works on values (not functions)
    """
    attrs = fetch.attrs()
    cf_iunits = cf_units.Unit(attrs[('var', 'units')])
    cf_ounits = cf_units.Unit(ounits)
    zo2 = cf_iunits.convert(_zero_one, cf_ounits)

    # y = mx + b slope & intercept
    b = zo2[0]    # b = y-intercept
    m = zo2[1] - zo2[0]    # m = slope

    attrs[('var', 'units')] = ounits
#    return attrdictx(
#        attrs = fetch.attrs,
#        data = fetch.data*m + b)
    return FetchTuple(fetch.attrs, fetch.data*m + b)

convert_unitsF = functional.lift()(convert_unitsV)


