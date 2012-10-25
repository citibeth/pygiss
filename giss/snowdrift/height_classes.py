import bisect
import giss.ncutil
import numpy as np

# Stuff to prepare conservation matricies and deal with height classes in Python.

# ----------------------------------------------------------------
def read_overlap_matrix(nc, var_name) :
	index_base = nc.variables[var_name + '.descr'].index_base
	index = giss.ncutil.read_ncvar(nc, var_name + '.index', dtype=np.int32) - index_base
	row = list(index[:,0])
	col = list(index[:,1])
	val = list(giss.ncutil.read_ncvar(nc, var_name + '.val', dtype=np.int32))

	return (row, col, val)

# ----------------------------------------------------------------
class HeightClassifier :
	def __init__(self, tops) :
		self.tops = tops
		self.nhc = tops.shape[0]
	def get_hclass(self, i1, elevation) :
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
	rows0 = overlap0[0]
	cols0 = overlap0[1]
	vals0 = overlap0[2]

	rows1 = []
	for ix in range(0, len(rows0)) :
		i1 = rows0[ix]
		i2 = cols0[ix]
		hc = height_classifier.get_hclass(i1, elevation2[i2])
		i1h = i1 * height_classifier.nhc + hc
		rows1.append(i1h)

	return (rows1, cols0, vals0)
# ----------------------------------------------------------------
# Conservation regions by height class, and 2x2 grid cell groups
def make_conserv2(overlap0, elevation2, height_classifier, nlon, nx=3, ny=3) :
	rows0 = overlap0[0]
	cols0 = overlap0[1]
	vals0 = overlap0[2]

	rows1 = []
	for ix in range(0, len(rows0)) :
		i1 = rows0[ix]
		ilat = i1 / nlon
		ilon = i1 - (ilat * nlon)

		i1new = (ilat/ny)*nlon + (ilon/nx)	# 2x2 grid groups
		i2 = cols0[ix]
		hc = height_classifier.get_hclass(i1, elevation2[i2])
		i1h = i1new * height_classifier.nhc + hc

		rows1.append(i1h)

	return (rows1, cols0, vals0)
