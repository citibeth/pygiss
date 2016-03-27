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
import numpy.ma as ma
import netCDF4
import os


def read_ncvar(nc, var_name, dtype='d') :
    """Reads a variable out of a scaled ACC file (netCDF format).

    Understands the missing_value attribute (or without it, just
    treates anything >1e10 as missing).

    Returns:    (Numpy Array)
        The variable.  Invalid values are set to np.nan

    """

    var = nc.variables[var_name]
    val = np.zeros(var.shape, dtype=dtype)
    val[:] = var[:]
#   val = var[:]


    if 'missing_value' in var.__dict__ :
        val[val == var.missing_value] = np.nan
    else :
        # Use generic cutoff
        val[np.abs(val) > 1.e10] = np.nan

    return val


# def read_ncvar(nc, var_name, mask_var = True) :
#   """Reads a variable out of a scaled ACC file (netCDF format).
# 
#   Understands the missing_value attribute (or without it, just
#   treates anything >1e10 as missing).
# 
#   Returns:    (np.ma.MaskedArray)
#       The variable.  Invalid values are masked
#       out.  Inside the data array, they are also set to np.nan
# 
#   """
# 
#   var = nc.variables[var_name]
#   val = var[:]
# 
#   # Mask the variable
#   if mask_var :
#       if 'missing_value' in var.__dict__ :
#           val = ma.masked_where(val == var.missing_value, val)
#       else :
#           # Use generic cutoff
#           val = ma.masked_where(np.abs(val) > 1.e10, val)
# 
#   # Eliminate masked values.  This way, things that don't understand
#   # masking (eg. snowdrift) won't get confused.
#   if ma.getmask(val) is not ma.nomask :
#       val.data[ma.getmask(val)] = np.nan
# 
# 
#   return val
# 
# --------------------------------------------------------
