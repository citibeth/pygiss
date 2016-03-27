# pyGISS: GISS Python Library
# Copyright (c) 2013 by Robert Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import netCDF4

# -------------------------------------------------------------
def _get_landmask(pism_nc) :
    """Converts the Pism land cover numbers into true/false for an ice mask.

    landcover:ice_sheet = 4 ;
    landcover:land = 2 ;
    landcover:local_ice_caps_not_connected_to_the_ice_sheet = 3 ;
    landcover:long_name = "Land Cover" ;
    landcover:no_data = 0 ;
    landcover:ocean = 1 ;
    landcover:standard_name = "land_cover" ;"""

    maskI = np.array(pism_nc.variables['mask'][:], dtype=np.int32)[0,:,:]
    maskI = np.where(maskI==2,np.int32(0),np.int32(1))
    return maskI
# -------------------------------------------------------------
def read_elevI_maskI(pism_fname) :
    """Reads elevI and maskI from a Pism data file
    Returns: (elevI, maskI) tuple
        elevI[n2] (np.array):
            Elevation of each ice grid cell (m)
        maskI[n2] (np.array, dtype=bool):
            False for ice grid cells, True for unused cells
    """

    # =============== Read stuff from ice grid (maskI, elevI)
#   print 'Opening ice data file %s' % pism_fname
    pism_nc = netCDF4.Dataset(pism_fname)

    # --- maskI
    maskI = _get_landmask(pism_nc)

    # --- elevI
    topg = np.array(pism_nc.variables['topg'][:], dtype='d')[0,:,:]
    thk = np.array(pism_nc.variables['thk'][:], dtype='d')[0,:,:]
    elevI = topg + thk

    pism_nc.close()
    return (elevI, maskI)
