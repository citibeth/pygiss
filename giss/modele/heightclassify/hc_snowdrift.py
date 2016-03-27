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

#!/usr/bin/env python
#
# Subroutines for adding height classes to existing ModelE input files.
# INPUT:
#     Global Initial Conditions file (from GIC in ModelE rundeck)
#     TOPO file (from TOPO in ModelE rundeck, GISS format)
#     Overlap matrix file
# OUTPUT:
#     Height-classified GIC file
#     Height-classified TOPO file (netCDF format)

import snowdrift
import giss.io.gissfile
import netCDF4
import giss.util
import giss.ncutil
import numpy as np
import odict
from _hcinput import *

# TODO: Re-do, for loadable interpolation matrices in the GCM

## Check length of a non-numpy variable
#def _check_len(var, llen, varname) :
#   if len(var) != llen :
#       raise Exception('%s(%d) should have length %d' % (varname, len(var), llen))

# ---------------------------------------------------------------
# The core subroutine that height-classifies variables in GIC and TOPO files
# @param height_max1h[n1 x nhc] Height class definitions
# @param overlap_fnames[nis = # ice sheets] Overlap matrix for each ice sheet
# @param elevation2[nis] DEM for each ice sheet
# @param masks2[nis] Ice landmask for each ice sheet
# @param ivars Variables needed to do height classification:
#        fgrnd1, flake1, focean1, fgice1, zatmo1, tlandi1, snowli1
def hc_vars_with_snowdrift(
# Parameters to be curried
height_max1h,
ice_sheet_descrs,   # [].{overlap_fname, elevation2, mask2}
# "Generic" parameters to remain
ivars) :

    # Fetch input variables from ivars
    fgrnd1 = ivars['fgrnd1']
    flake1 = ivars['flake1']
    focean1 = ivars['focean1']
    fgice1 = ivars['fgice1']
    zatmo1 = ivars['zatmo1']
    tlandi1 = ivars['tlandi1']
    snowli1 = ivars['snowli1']

    # Get dimensions
    nhc = height_max1h.shape[0]
    n1 = height_max1h.shape[1]

    # Check dimensions
    check_shape(fgice1, (n1,), 'fgice1')
    check_shape(fgrnd1, (n1,), 'fgrnd1')
    check_shape(zatmo1, (n1,), 'zatmo1')
    check_shape(tlandi1, (n1,2), 'tlandi1')
    check_shape(snowli1, (n1,), 'snowli1')
    check_shape(height_max1h, (nhc,n1), 'height_max1h')
    for descr in ice_sheet_descrs :
        print descr.__dict__
        check_shape(descr.elevation2, (descr.n2,), '%s:elevation2' % descr.overlap_fname)
        check_shape(descr.mask2, (descr.n2,), '%s:mask2' % descr.overlap_fname)

    # Make height-classified versions of vars by copying
    o_tlandi1h = np.zeros((nhc,n1,2))
    o_snowli1h = np.zeros((nhc,n1))
    o_elev1h = np.zeros((nhc,n1))
    for ihc in range(0,nhc) :
        o_tlandi1h[ihc,:] = tlandi1[:]
        o_snowli1h[ihc,:] = snowli1[:]
        o_elev1h[ihc,:] = zatmo1[:]

    # Initialize fhc1h to assign full weight to first height class
    o_fhc1h = np.zeros((nhc,n1))
    o_fhc1h[:] = 0
    o_fhc1h[0,:] = 1

    # Loop over each ice sheet
    for descr in ice_sheet_descrs :
        # Load the overlap matrix
        sd = snowdrift.Snowdrift(descr.overlap_fname)
        sd.init(descr.elevation2, descr.mask2, height_max1h)
        if sd.grid1().n != n1 :
            raise Exception('%s:grid1[%d] should have dimension %d', (fname, sd.grid1().n, n1))
        if sd.grid2().n != descr.n2 :
            raise Exception('%s:grid2[%d] should have dimension %d', (fname, sd.grid2().n, descr.n2))

        # Use it to compute useful stuff
        sd.compute_fhc(o_fhc1h, o_elev1h, fgice1)

    # ======= Adjust fgrnd accordingly, to keep (FGICE + FGRND) constant
    o_fgrnd1 = np.ones(n1) - focean1 - flake1 - fgice1

    # Compute zatmo, overlaying file from disk
    x_zatmo1 = np.nansum(o_fhc1h * o_elev1h, 0)
    mask = np.logical_not(np.isnan(x_zatmo1))
    o_zatmo1 = np.zeros(n1)
    o_zatmo1[:] = zatmo1[:]
    o_zatmo1[mask] = x_zatmo1[mask]

    # Return result
    ovars = odict.odict({
        'tlandi1h' : o_tlandi1h,
        'snowli1h' : o_snowli1h,
        'elev1h' : o_elev1h,
        'fhc1h' : o_fhc1h,
        'fgrnd1' : o_fgrnd1,
        'zatmo1' : o_zatmo1})
    return ovars
