import bisect
import giss.ncutil
import numpy as np
import scipy.sparse
import pyproj

# Stuff to prepare conservation matricies and deal with height classes in Python.

# ----------------------------------------------------------------
# Read a pair of (proj, llproj) out of a netCDF Overlap file
# @param var_name (eg: grid1)
def read_projs(nc, var_name) :
	info_name = var_name + '.info'
	sproj = str(nc.variables[info_name].projection)
	print 'sproj = %s' % sproj
	sllproj = str(nc.variables[info_name].latlon_projection)
	print 'sllproj = %s' % sllproj
	projs = (pyproj.Proj(sllproj), pyproj.Proj(sproj))	# src & destination projection
	return projs

# ----------------------------------------------------------------
def read_sparse_matrix(nc, var_name) :
	descr_var = nc.variables[var_name + '.descr']
	index_base = descr_var.index_base
	index = giss.ncutil.read_ncvar(nc, var_name + '.index', dtype=np.int32) - index_base
	row = index[:,0]
	col = index[:,1]
	val = giss.ncutil.read_ncvar(nc, var_name + '.val', dtype='d')

	return scipy.sparse.coo_matrix((val, (row, col)), shape=(descr_var.nrow, descr_var.ncol))
#	return (row, col, val)

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
