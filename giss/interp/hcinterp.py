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

#import giss.snowdrift
#import netCDF4
import numpy as np
#import giss.ncutil
#import pyproj
import scipy.sparse
import operator
import bisect
import sys
from giss.interp import interp

class HCInterp(interp.Interp):

	"""Initializes, stores and applies an interpolation matrix to go from
	GCM to ice grid, interpolating in elevation space only.  If you wish to
	interpolate in X-Y as well, see BilinInterp.

	Definitions:
		(see Interp)

	Attributes:
		(see Interp)
	"""

	def __init__(self, overlap, _elev1h, _elev2, _mask2) :
	"""Construct the elevation-only interpolation matrix

	Args:
		overlap[n1, n2] (scipy.sparse.coo_matrix):
			The overlap matrix between grid1 (GCM) and grid2 (ice).
		_elev1h[nhc, n1] (np.array):
			Set of elevation points in each grid cell we're computing on.
			Frequently, elevation points are the same for all grid cells.
			(may be any shape, as long as shape[0]=nhc and shape[1:] = n1)
		_elev2[n2] (np.array):
			Elevation of each ice grid cell (or grid point)
			(may be any shape, as long as it has n2 elements)
		_mask2[n2] (np.array, dtype=bool):
			True for gridcells in ice grid that have ice.
			(may be any shape, as long as it has n2 elements)
			NOTE: The sense of this mask is OPPOSITE that used in numpy.ma
	"""

		# Reshape arrays to our preferred version
		nhc = _elev1h.shape[0]
		n1 = reduce(operator.mul, _elev1h.shape[1:])
		elev1h = _elev1h.reshape((nhc, n1))
		n2 = reduce(operator.mul, _elev2.shape)
		elev2 = _elev2.reshape(n2)
		mask2 = _mask2.reshape(n2)
		self.mask2 = mask2

		# 
		# overlap = giss.snowdrift.read_sparse_matrix(overlap_nc, 'overlap')

		# Remove rows and columns from the overlap matrix according to mask1h and mask2
		# include eliminating rows based on nans in elev1h or elevation2
		rows2 = []
		cols2 = []
		vals2 = []
		for (i1, i2, val) in zip(overlap.row, overlap.col, overlap.data) :
			if not mask2[i2] : continue
			rows2.append(i1)
			cols2.append(i2)
			vals2.append(val)
		overlapb = scipy.sparse.coo_matrix((vals2, (rows2, cols2)), shape=overlap.shape)

		# Sum columns of matrix
		np.set_printoptions(threshold='nan')
		sum_by_col = np.array(overlapb.sum(axis=0)).reshape(-1)
#		print sums.shape
#		print type(sums)
#		sys.exit(0)

		# The matrix we're making
		orow = []	# 0..nhc*n1
		ocol = []	# 0..n2
		oval = []

		# Iterate through our matrix
		for (i1, i2, overlap_area) in zip(rows2, cols2, vals2) :
			overlap_ratio = overlap_area / sum_by_col[i2]
			elevs = elev1h[:,i1]
			elevation = np.max(elev2[i2], 0.0)
			ihcb = bisect.bisect_left(elevs, elevation)
			if ihcb > 0 :	# Interpolate between two points
				ihca = ihcb - 1
				ratio = (elevs[ihcb] - elevation) / (elevs[ihcb] - elevs[ihca])
				ocol.extend([i2, i2])
				orow.extend([ihca*n1 + i1, ihcb*n1 + i1])
				oval.extend([overlap_ratio * ratio, overlap_ratio * (1.0-ratio)])
#				print sum_by_col[i2]
			else :		# Extrapolate off of left hand side of the function
				# slope = (vb-va)/(eb-ea)
				# val = va - (ea - elevation) * slope
				ihcb = 1
				ihca = 0
				ratio = (elevs[ihca] - elevation) / (elevs[ihcb] - elevs[ihca])
				ocol.extend([i2,i2])
				orow.extend([ihca*n1 + i1, ihcb*n1 + i1])
				oval.extend([overlap_ratio * (1.0+ratio), -overlap_ratio * ratio])

		self.M = scipy.sparse.coo_matrix((oval, (orow, ocol)), shape=(nhc*n1, n2))
