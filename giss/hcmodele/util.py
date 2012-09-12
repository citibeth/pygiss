import numpy as np
import netCDF4

# Grabs grid1.size() out of a grid overlap file
def get_grid1_size(overlap_fname) :
	nc = netCDF4.Dataset(overlap_fname, 'r')
	v = nc.variables['grid1.info']
	return v.max_index - v.index_base + 1
# -------------------------------------------------------------------

# Creates a height_max1h field that is the same for all grid cells
# @param n1 Number of GCM grid cells
# @param tops Top of each height class, nhc == len(tops)
def const_height_classes(tops, n1) :
	return np.tile(np.array(tops), (n1,1))		# Produces an n1 x nhc array
# -------------------------------------------------------------------
