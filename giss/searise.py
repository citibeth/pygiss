import numpy as np
import netCDF4

# -------------------------------------------------------------
def _get_landmask(searise_nc) :
	"""Converts the SeaRise land cover numbers into true/false for an ice mask.

	landcover:ice_sheet = 4 ;
	landcover:land = 2 ;
	landcover:local_ice_caps_not_connected_to_the_ice_sheet = 3 ;
	landcover:long_name = "Land Cover" ;
	landcover:no_data = 0 ;
	landcover:ocean = 1 ;
	landcover:standard_name = "land_cover" ;"""

	mask2 = np.array(searise_nc.variables['landcover'], dtype=np.int32).flatten('C')
	mask2 = np.where(mask2==4,np.int32(1),np.int32(0))
	return mask2
# -------------------------------------------------------------
def read_elevation2_mask2(searise_fname) :
	"""Reads elevation2 and mask2 from a SeaRISE data file
	Returns: (elevation2, mask2) tuple
		elevation2[n2] (np.array):
			Elevation of each ice grid cell (m)
		mask2[n2] (np.array, dtype=bool):
			True for ice grid cells, false for unused cells
	"""

	# =============== Read stuff from ice grid (mask2, elevation2)
	print 'Opening ice data file %s' % searise_fname
	searise_nc = netCDF4.Dataset(searise_fname)

	# --- mask2
	mask2 = _get_landmask(searise_nc)

	# --- elevation2
	topg = np.array(searise_nc.variables['topg'], dtype='d').flatten('C')
	thk = np.array(searise_nc.variables['thk'], dtype='d').flatten('C')
	elevation2 = topg + thk

	searise_nc.close()
	return (elevation2, mask2)
