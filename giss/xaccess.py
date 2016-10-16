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
@functional.function
def ncattrs(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    """Produces extended attributes on a variable fetch operation"""
    nc = ncopen(fname)

    attrs = {('var', key) : val
        for key,val in nc.variables[var_name].__dict__.items()}

    attrs[('fetch', 'file')] = fname
    attrs[('fetch', 'var')] = var_name
    attrs[('fetch', 'index')] = index
    attrs[('fetch', 'nan')] = np.nan
    attrs[('fetch', 'missing_value')] = missing_value
    attrs[('fetch', 'missing_threshold')] = missing_threshold

    return functional.WrapCombine(attrs, functional.intersect_dicts)
# --------------------------------------------

