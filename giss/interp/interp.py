import numpy as np

class Interp:
	"""Superclass for other items in this package.

	Definitions:
		grid1 = GCM grid.  It must be a lat/lon grid, at relatively lower resolution.
			n1lon = Number of grid cells in longitude direction
			n1lat = Number of grid cells in latitude direction
			n1 = # grid cells in grid1 = n1lat * n1lon
		grid2 = ice grid.  It can be any kind of grid or mesh, at higher resolution.
			Assumes a zero-order paramterization (constant values in each grid cell).
			n2 = # grid cells in grid2
		nhc = Number of elevation classes

	Attributes:
		M[n1, n2] (scipy.sparse.coo_matrix):
			The matrix used to interpolate.  In general, regrid from
			grid1 (GCM) to grid2 (ice) using
				v2 = bilinz.transpose() * v1
		mask2[n2] (np.array, dtype=bool):
			True for gridcells in ice grid that have ice.
			NOTE: The sense of this mask is OPPOSITE that used in numpy.ma
	"""

	def interpolate(self, _val1h) :
		"""Apply the interpolation matrix.
		Args:
			_val1h[n1] (np.array):
				Field on grid1 to interpolate/regrid.
				Unused values should be set to np.nan
				May have any number of dimensions, as long as total number of elements is n1

		Returns:
			val2[n2] (np.array):
				The same field, regridded to grid2.
				== bilinz.transpose() * _val1h, plus masking/NaN issues.
				Cells masked out in grid2 are set to np.nan
		"""

		val1h = _val1h.reshape(reduce(operator.mul, _val1h.shape))	# Convert to 1-D
		v1h = np.copy(val1h)
		v1h[np.isnan(val1h)] = 0	# Prepare for multiply
		val2 = self.M.transpose() * v1h
		val2[np.logical_not(self.mask2)] = np.nan
		return val2

