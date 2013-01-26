import giss.proj
import numpy as np
import pyproj
import operator
import numpy.ma as ma

class ProjXYPlotter :
	"""A plotter for Cartesian-gridded data projected onto the sphere.
	(Typically used to plot data on the ice grid)

	Plotters provide a pcolormesh() subroutine that abstracts away
	the specifics of the grid used.

	"""
	
	def __init__(self, xb2, yb2, sproj):
		"""Construct the plotter.

		Args:
			xb2[nx]:
				Boundaries of grid cells in the X direction
			yb2[nx]:
				Boundaries of grid cells in the Y direction
			sproj (string):
				Proj.4 string for the projection used to map between Cartesian and the globe.

		Attributes:
			nx2[], ny2[]:
				Number of grid cells in X and Y directions
			n2:
				Total number of grid cells
			mesh_lons[n2], mesh_lats[n2]:
				Lat/lon of every grid cell boundary intersection.
		"""

		self.nx2 = len(xb2)-1
		self.ny2 = len(yb2)-1

		# Check dims
		self.n2 = self.nx2 * self.ny2

		# Get the ice grid projection as well, so we can convert our local
		# mesh to lat/lon
		self.sproj = sproj
		(llproj, xyproj) = giss.proj.make_projs(sproj)

		# Create a quadrilateral mesh in X/Y space
		xs, ys = np.meshgrid(xb2, yb2)

		# Transform it to lat/lon space
		self.mesh_lons, self.mesh_lats = pyproj.transform(xyproj, llproj, xs, ys)


	# @param basemap the basemap instance we wish to plot on
	# @param _val2 a Numpy array or masked array with the values to plot
	def pcolormesh(self, basemap, _val2, **plotargs) :
		"""
		Args:
			basemap:
				The map to plot on.
			_val2[n2] (np.array):
				The value to plot (preferably np.ma.MaskedArray)
				May be any shape, as long as it has n2 elements.
		See:
			pcolormesh()"""

		n2 = reduce(operator.mul, _val2.shape)
		if self.n2 != n2 :
			raise Exception("Plotter's n2 (%d) != data's n2 (%d)" % (self.n2, n2))

		# Convert to map space
		mesh_mapx, mesh_mapy = basemap(self.mesh_lons, self.mesh_lats)

		# Reshape back to xy
		# Careful of dimension order, it needs to match dims in the quadrilateral mesh
		val2xy = _val2.reshape((self.ny2, self.nx2))

		if not issubclass(type(val2xy), ma.MaskedArray) :
			# Not a masked array, mask out invalid values (eg NaN)
			val2xy = ma.masked_invalid(val2xy)

		# Plot our result using the quadrilateral mesh
		return basemap.pcolormesh(mesh_mapx, mesh_mapy, val2xy, **plotargs)

