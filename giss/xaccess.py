import functools
from giss import bind,checksum,memoize,functional
import collections
import numpy as np
import netCDF4

"""Generalized functional-style access to data."""

# -------------------------------------------------------------

# -------------------------------------------------------------
# Allows us to do indexing without an array to index ON.
#   We can say:   index = _ix[4:5,:]
class IndexClass(object):
    """Extract slice objects"""
    def __getitem__(self, t):
        if isinstance(t, tuple):
            return t
        return (t,)

# Singleton allows us to say, eg: ix[2,3,4:5]
_ix = IndexClass()
# -------------------------------------------------------------

@memoize.local()
def ncopen(name):
    nc = netCDF4.Dataset(name, 'r')
    return nc

# ---------------------------------------------------------------
@functional.thunkify
def ncdata(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    """Simple accessor function for data in NetCDF files.
    Ops on this aren't very interesting because it is a
    fully-bound thunk."""

    nc = ncopen(fname)
    var = nc.variables[var_name]

    data = var[index]
    if missing_value is not None:
        # User override of NetCDF standard
        data[val == missing_value] = nan
    elif hasattr(var, 'missing_value'):
        # NetCDF standard
        data[val == var.missing_value] = nan
    elif missing_threshold is not None:
        # Last attempt to fix a broken file
        data[np.abs(val) > missing_threshold] = nan

    return data
# --------------------------------------------
NCFetchTuple = functional.NamedTuple('NCFetchTuple', ('attrs', 'data'))

@functional.function
def ncfetch(file_name, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
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

    attrs[('fetch', 'file_name')] = file_name
    attrs[('fetch', 'var_name')] = var_name
    attrs[('fetch', 'index')] = index
    attrs[('fetch', 'nan')] = np.nan
    attrs[('fetch', 'missing_value')] = missing_value
    attrs[('fetch', 'missing_threshold')] = missing_threshold

    return NCFetchTuple(
        functional.WrapCombine(attrs, functional.intersect_dicts),
        ncdata(file_name, var_name, *index, nan=nan,
            missing_value=missing_value,
            missing_threshold=missing_threshold))

# --------------------------------------------

