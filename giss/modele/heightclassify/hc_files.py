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

import snowdrift
import giss.io.gissfile
import netCDF4
import giss.util
import giss.ncutil
import numpy as np
import odict
from _hcinput import *

## Check length of a non-numpy variable
#def _check_len(var, llen, varname) :
#   if len(var) != llen :
#       raise Exception('%s(%d) should have length %d' % (varname, len(var), llen))

# ----------------------------------------------------------
def hc_files(TOPO_iname, GIC_iname, TOPO_oname, GIC_oname, hc_vars, nhc) :

    """Takes a TOPO and GIC file set up for a
    non-height-classified run of ModelE, and creates new TOPO and
    GIC files suitable for use with heigth classes / elevation points.

    Args:
        TOPO_iname (string, GISS-format):
            Name of input TOPO file
        GIC_iname (string, netCDF):
            Name of input GIC file  
        TOPO_oname (string, netCDF):
            Name of output TOPO file
        GIC_oname (string, netCDF):
            Name of output GIC file
        hc_vars (function):
            Function to do the work of height-classifying the
            variables in the GIC and TOPO files.  See hc_snowdrift.py
            and hc_cesm.py
        nhc (int):
            Number of elevation classes to set up"""

    ituples = {}
    ivars = {}

    # ================ Read the TOPO file and get dimensions from it
    topo = _read_all_giss_struct(TOPO_iname)
    jm = topo['zatmo'].val.shape[0]     # int
    im = topo['zatmo'].val.shape[1]     # int
    n1 = jm * im
    s_jm_im = topo['zatmo'].sdims       # symbolic
    s_nhc_jm_im = ('nhc',) + s_jm_im    # symbolic


    # ================ Prepare the input variables
    # ...from the TOPO file
    for var in ('fgrnd', 'flake', 'focean', 'fgice', 'zatmo') :
        tuple = topo[var]
        ituples[var] = tuple
        ivars[var + '1'] = tuple.val.reshape((n1,))

    # ...from the GIC file
    ncgic = netCDF4.Dataset(GIC_iname, 'r')
    for var in ('tlandi', 'snowli') :
        tuple = _read_ncvar_struct(ncgic, var, float)
        ituples[var] = tuple
        print tuple.name,n1,str(tuple.val.shape)
        new_dims = (n1,) + tuple.val.shape[2:]
        ivars[var + '1'] = tuple.val.reshape(new_dims)
    # Leave this file open, we'll need it later whne we copy out all vars

    # ================= Height-classify the variables
    ovars = hc_vars(ivars)

    # ============= Set up new variables to output as tuples
    otlist = [
        # (name, val, sdims, dtype)
        ('tlandi',
            ovars['tlandi1h'].reshape((nhc,jm,im,2)),
            ('nhc',) + ituples['tlandi'].sdims,
            ituples['tlandi'].dtype),
        ('snowli',
            ovars['snowli1h'].reshape((nhc,jm,im)),
            s_nhc_jm_im, ituples['snowli'].dtype),
        ('elevhc',
            ovars['elev1h'].reshape((nhc,jm,im)),
            s_nhc_jm_im, ituples['zatmo'].dtype),
        ('fhc',
            ovars['fhc1h'].reshape((nhc,jm,im)),
            s_nhc_jm_im, 'f8'),
        ('fgrnd',
            ovars['fgrnd1'].reshape((jm,im)),
            s_jm_im, ituples['fgrnd'].dtype),
        ('zatmo',
            ovars['zatmo1'].reshape((jm,im)),
            s_jm_im, ituples['zatmo'].dtype)]

    # Convert to a dict of tuple Structs
    otuples = odict.odict()
    for ot in otlist :
        otuples[ot[0]] = giss.util.Struct({
            'name' : ot[0], 'val' : ot[1],
            'sdims' : ot[2], 'dtype' : ot[3]})

    # ============= Write the TOPO file (based on old variables)
    # Collect variables to write
    otuples_remain = odict.odict(otuples)
    wvars = []  # Holds pairs (source, name)
    for name in topo.iterkeys() :
        # Fetch variable from correct place
        if name in otuples :
            wvars.append((otuples, name))
            del otuples_remain[name]
        else :
            wvars.append((topo, name))

    # Define and write the variables
    write_netcdf(TOPO_oname, wvars)


    # ============= Write out new GIC
    # Collect variables to write (from original GIC file)
    wvars = []  # Holds pairs (source, name)
    for name in ncgic.variables :
        # Fetch variable from correct place
        if name in otuples :
            wvars.append((otuples, name))
            del otuples_remain[name]
        else :
            wvars.append((ncgic, name))

    # Add on new variable(s) we've created
    for name in otuples_remain.iterkeys() :
        wvars.append((otuples, name))

    # Define and write the variables
    write_netcdf(GIC_oname, wvars)

