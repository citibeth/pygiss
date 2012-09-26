import snowdrift
import numpy as np
import netCDF4
import operator
import giss.ncutil
import pyproj
import numpy.ma as ma

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
#	return np.tile(np.array(tops), (n1,1))		# Produces an n1 x nhc array
	return np.tile(np.array(tops, ndmin=2).transpose(), (1,n1))

# -------------------------------------------------------------------

# ==============================================================
# Used for plotting data living on the ice grid (if the ice grid is Cartesian)
class Grid2Plotter_XY :
	def __init__(self, overlap_fname) :
		self.overlap_fname = overlap_fname

		# ======= Read our own grid2 info from the overlap file
		# Assumes an XY grid for grid2
		nc = netCDF4.Dataset(overlap_fname, 'r')
		if nc.variables['grid2.info'].__dict__['type'] != 'xy' :
			raise Exception('Expecting grid2.info:type == "xy"')
		xb2 = giss.ncutil.read_ncvar(nc, 'grid2.x_boundaries')
		self.nx2 = len(xb2)-1
		yb2 = giss.ncutil.read_ncvar(nc, 'grid2.y_boundaries')
		self.ny2 = len(yb2)-1

		# Check dims
		self.n2 = self.nx2 * self.ny2

		# Get the ice grid projection as well, so we can convert our local
		# mesh to lat/lon
#		print nc.variables['grid2.info'].__dict__
		sproj = str(nc.variables['grid1.info'].getncattr('projection'))
		sllproj = str(nc.variables['grid1.info'].getncattr('latlon_projection'))
		projs = (pyproj.Proj(sllproj), pyproj.Proj(sproj))	# src & destination projection

		# Create a quadrilateral mesh in X/Y space
		xs, ys = np.meshgrid(xb2, yb2)

		# Transform it to lat/lon space
		self.mesh_lons, self.mesh_lats = pyproj.transform(projs[1], projs[0], xs, ys)

	# @param mymap the basemap instance we wish to plot on
	# @param val2_varshape a Numpy array or masked array with the values to plot
	def pcolormesh(self, mymap, val2_varshape, **plotargs) :
#		print 'max(val2_varshape) = %f' % np.max(val2_varshape)

		n2 = reduce(operator.mul, val2_varshape.shape)
		if self.n2 != n2 :
			raise Exception("Plotter's n2 (%d) != data's n2 (%d)" % (self.n2, n2))

		# Convert to map space
		mesh_mapx, mesh_mapy = mymap(self.mesh_lons, self.mesh_lats)

		# Reshape back to xy
		# Careful of dimension order, it needs to match dims in the quadrilateral mesh
		val2xy = val2_varshape.reshape((self.ny2, self.nx2))

		# Plot our result using the quadrilateral mesh
		return mymap.pcolormesh(mesh_mapx, mesh_mapy, val2xy, **plotargs)


# =====================================================================

# (elevation2, mask2) can come from giss.searise.read_elevation2_mask2
# @param mask2 a Boolean array, True where we want landice, False where none
#        (This is the opposite convention from numpy masked arrays)
# @param height_max1h (nhc x n1) array of height class definitions
class Grid1hPlotter :
	def __init__(self, grid2_plotter, height_max1h, elevation2, mask2) :
		self.grid2_plotter = grid2_plotter
		self.sd = snowdrift.Snowdrift(grid2_plotter.overlap_fname)
		self.sd.init(elevation2, mask2, height_max1h)
		self.mask2 = mask2

		# Check dims
		if grid2_plotter.n2 != self.sd.grid2().n :
			raise Exception('n2 (%d) != sd.grid2().n (%d)' % (grid2_plotter.n2, sd.grid2().n))

	def pcolormesh(self, mymap, val1h_varshape, **plotargs) :
		# Consolidate dimensions so this is (nhc, n1)
		nhc = val1h_varshape.shape[0]
		n1 = reduce(operator.mul, val1h_varshape.shape[1:])
		val1h = val1h_varshape.reshape((nhc, n1))

		if self.sd.grid1().n != n1 :
			raise Exception('sd.grid1().n (%d) != n1 (%d)' % (sd.grid1().n, n1))

		# Do a simple regrid to grid 2 (the local / ice grid)
		val2 = np.zeros((self.grid2_plotter.n2,))
		val2[:] = np.nan
		self.sd.downgrid(val1h, val2, merge_or_replace = 'replace', correct_proj_area = 0)	# Area-weighted remapping

		# Masked areas will remain nan in val2
		# Create a numpy plotting mask with this in mind.
		#val2_masked = ma.masked_array(val2, np.isnan(val2))
		val2_masked = ma.masked_invalid(val2)

		# Plot using our local plotter
		return self.grid2_plotter.pcolormesh(mymap, val2_masked, **plotargs)
