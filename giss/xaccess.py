import functools
from giss import bind,checksum,memoize,functional
import collections
import numpy as np
import netCDF4

"""Generalized functional-style access to data."""

class IndexClass(object):
    """Extract slice objects"""
    def __getitem__(self, t):
        if isinstance(t, tuple):
            return t
        return (t,)

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
        return real_fn

def intersect_dicts(a,b):
    return {key : a[key] \
        for key a.keys()&b.keys() \
        if a[key] == b[key]}

# Functions that return meta-data
class AttrsOps(object):
    def __add__(self, other):
        def real_fn(*args, **kwargs):
            sret = self(*args, **kwargs)
            oret = other(*args, **kwargs)
            return intersect_dicts(sret, oret)

# Functions that return tuples of things
class MultiOps(object):
    def __add__(self, other):
        def real_fn(*args, **kwargs):
            sret = self(*args, **kwargs)
            oret = other(*args, **kwargs)
            return tuple(s + o for s,o in zip(sret, oret))
# --------------------------------------------
@functional.addops(ArrayOps)
def ncdata(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    """Simple accessor function for data in NetCDF files.
    Returns: (attrs, thunk)
        attrs:
            Attributes of the NetCDF variable
        thunk:
            thunk() returns the data."""

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
@functional.addops(AttrOps)
def ncattrs(fname, var_name):
    """Fetches attributes of a variable."""
    nc = ncopen(fname)
    return dict(nc.variables[var_name]__dict__)

@functional.addops(MultiOps)
def ncfetch(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    attrs = ncattrs(fname, var_name)
    attrs['file'] = fname
    attrs['var'] = var_name
    attrs['index'] = index
    attrs['nan'] = np.nan
    attrs['missing_value'] = missing_value
    attrs['missing_threshold'] = missing_threshold
    return (attrs,
        functional.bind(ncdata, fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None))

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

