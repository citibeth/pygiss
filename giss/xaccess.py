import functools
from giss import bind,checksum,memoize,functional
import collections
import numpy as np
import netCDF4

"""Generalized functional-style access to data."""

class IndexClass(object):
    """Extract slice objects"""
    def __getitem__(self, tuple):
        return tuple
# Singleton allows us to say, eg: ix[2,3,4:5]
_ix = IndexClass()

@memoize.local()
def ncopen(name):
    return netCDF4.Dataset(name, 'r')

# -------------------------------------------------------------
# Data access functions return Numpy Array
class ArrayOps(object):
    def __add__(self, other):
        def real_fn(*args, **kwargs):
            return self(*args, **kwargs) + other(*args, **kwargs)

@functional.addops(ArrayOps)
def ncdata(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    """Simple accessor function for data in NetCDF files"""
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

def ncattr(fname, var_name, attr_name):
    nc = ncopen(fname)
    if var_name is None:
        return getattr(nc, attr_name)
    else:
        return getattr(nc.variables[var_name], attr_name)
# ------------------------------------------
# Higher-order functions
def sum(*funcs):
    def realfn(*args, **kwargs):
        val = funcs[0](*args, **kwargs)
        for fn in funcs[1:]:
            val1 = fn(*args, **kwargs)
            val += val1
        return val
    return realfn

