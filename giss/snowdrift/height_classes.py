import bisect
import giss.ncutil
import numpy as np
import scipy.sparse
import pyproj

# Stuff to prepare conservation matricies and deal with height classes in Python.


# ----------------------------------------------------------------
class HeightClassifier :
	"""HeightClassifier converts elevation to elevation class, for a given GCM grid cell.
	NOTE: This doesn't really make sense for elevation points."""

	def __init__(self, tops) :
		"""Args:
			tops (np.array): Top of each elevation class (m).
			Maybe be one of two shapes:
				tops[nhc]: Use same elevation classes for all grid cells
				tops[nhc, n1]: Use different elevation classes for each grid cell
		"""

		self.tops = tops
		self.nhc = tops.shape[0]
	def get_hclass(self, i1, elevation) :
	"""Given an elevation and a GCM grid cell index, tells you the elevation class.
	Args:
		i1 (int):
			Index of the GCM grid cell (0-based)
		elevation:
			Elevation to convert (m)
	Returns:	(int)
		Elevation class of <elevation> in grid cell <i1>.
	"""
		# Get tops array just for this grid cell
		if len(self.tops.shape) == 2 :
			i1tops = self.tops[:,i1]
		else :
			i1tops = self.tops

		# Binary search
		ret = bisect.bisect_left(i1tops, elevation)
		return ret
# ----------------------------------------------------------------
def height_classify_overlap(overlap0, elevation2, height_classifier) :
	"""Height-classify an overlap matrix.
	That is... given an overlap matrix between the regular GCM grid and the ice grid,
	and given a set of elevation classes, compute the overlap matrix between the
	height-classified GCM grid and the ice grid.

	Args:
		overlap0 (scipy.sparse.coo_matrix):
			The original overlap matrix, as read from Snowdrift netCDF file.
		elevation2[n2] (np.array):
			Elevation of each ice grid cell
		height_classifier (HeightClassifier):
			Used to compute elevation classes from elevations.
	Returns:	(scipy.sparse.coo_matrix)
		The height-classified overlap matrix."""

	rows0 = overlap0.rows
	cols0 = overlap0.cols
	vals0 = overlap0.data

	rows1 = []
	for ix in range(0, len(rows0)) :
		i1 = rows0[ix]
		i2 = cols0[ix]
		hc = height_classifier.get_hclass(i1, elevation2[i2])
		i1h = i1 * height_classifier.nhc + hc
		rows1.append(i1h)

	n1 = overlap0.shape[0]
	n2 = overlap0.shape[1]
	nhc = height_classifier.nhc

	return scipy.sparse.coo_matrix((vals0, (rows1, cols0)), shape=(n1*nhc, n2))
# ----------------------------------------------------------------
# Conservation regions by height class, and 2x2 grid cell groups
def make_conserv2(overlap0, elevation2, height_classifier, nlon, nx=3, ny=3) :
	nhc = height_classifier.nhc
	n1 = overlap0.shape[0]
	n2 = overlap0.shape[1]

	row1 = np.zeros(overlap0.row.shape, dtype=overlap0.row.dtype)
	for ix in range(0, len(overlap0.row)) :
		i1 = overlap0.row[ix]
		ilat = i1 / nlon
		ilon = i1 - (ilat * nlon)

		i1new = (ilat/ny)*nlon + (ilon/nx)	# 2x2 grid groups
		i2 = overlap0.col[ix]
		hc = height_classifier.get_hclass(i1, elevation2[i2])
		i1h = i1new * nhc + hc

		row1[ix] = i1h

	return scipy.sparse.coo_matrix((overlap0.data, (row1, overlap0.col)), shape=(n1*nhc, n2))
