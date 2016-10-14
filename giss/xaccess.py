import functools
from giss.bind import *

"""Generalized functional-style access to data."""


def ncvar(nc, var_name, *index):
    """Simple accessor function for NetCDF files"""
    return nc.variables[var_name][index]

