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

#import snowdrift
#import giss.io.gissfile
#import netCDF4
#import giss.util
#import giss.ncutil
import numpy as np
import hc_snowdrift
#import odict
#from hcinput import *


# TODO: Re-do, for loadable interpolation matrices in the GCM

# ---------------------------------------------------------------
# The core subroutine that height-classifies variables in GIC and TOPO files
# @param height_max1h[n1 x nhc] Height class definitions
# @param overlap_fnames[nis = # ice sheets] Overlap matrix for each ice sheet
# @param elevation2[nis] DEM for each ice sheet
# @param masks2[nis] Ice landmask for each ice sheet
# @param ivars Variables needed to do height classification:
#        fgrnd1, flake1, focean1, fgice1, zatmo1, tlandi1, snowli1
def hc_vars_ncar(
# Parameters to be curried
height_max1h,
ice_sheet_descrs,   # [].{overlap_fname, elevation2, mask2}
# "Generic" parameters to remain
ivars) :


    ovars = hc_snowdrift.hc_vars_with_snowdrift(height_max1h, ice_sheet_descrs, ivars)

    elev1h = ovars['elev1h']
    elev1h[0,:] = height_max1h[0,:] * .5
    for ihc in range(1,elev1h.shape[0]) :
        elev1h[ihc:,:] = (height_max1h[ihc,:] + height_max1h[ihc-1,:]) * .5

    fhc1h = ovars['fhc1h']
        fhc1h[fhc1h == 0] = 1e-30

    return ovars


