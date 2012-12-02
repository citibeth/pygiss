#import giss.snowdrift
#import netCDF4
import numpy as np
#import giss.ncutil
#import pyproj
import scipy.sparse
import operator
import bisect
import sys

class HCInterp:
	def interpolate(self, _val1h) :
		val1h = _val1h.reshape(reduce(operator.mul, _val1h.shape))	# Convert to 1-D
		v1h = np.copy(val1h)
		v1h[np.isnan(val1h)] = 0	# Prepare for multiply
		val2 = self.M.transpose() * v1h
		val2[np.logical_not(self.mask2)] = np.nan
		return val2

	# @param overlap Read from netCDF file with giss.snowdrift.read_sparse_matrix()
	def __init__(self, overlap, _elev1h, _elev2, _mask2) :
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
