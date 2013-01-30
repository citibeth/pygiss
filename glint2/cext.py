import _glint2
import scipy.sparse
import numpy as np

# Interface with C++ extension

# Pass-through class to glint2 module
Grid = _glint2.Grid

def _coo_to_tuple(coo) :
	return (coo._shape[0], coo._shape[1],
		coo.row, coo.col, coo.data)

def _tuple_to_coo(tuple) :
	nrow1 = tuple[0]
	ncol1 = tuple[1]
	rows1 = tuple[2]
	cols1 = tuple[3]
	data1 = tuple[4]
	return scipy.sparse.coo_matrix((data1, (rows1, cols1)), shape=(nrow1, ncol1))

# -------------------------------------------------------
def height_classify(overlap, elev2, hcmax) :
	tret = _glint2.height_classify(_coo_to_tuple(overlap), elev2, hcmax)
	return _tuple_to_coo(tret)

def hp_interp(overlap, elev2, hpdefs) :
	tret = _glint2.hp_interp(_coo_to_tuple(overlap), elev2, hpdefs)
	return _tuple_to_coo(tret)

# Puts A*x into y, does not overwrite unused elements of y
# @param yy OUTPUT
def coo_matvec(coomat, xx, yy) :
	yy = yy.reshape(-1)
	xx = xx.reshape(-1)

	_glint2.coo_matvec(_coo_to_tuple(coomat), xx, yy)
	return

def coo_multiply(coomat, xx, fill=np.nan) :
	xx = xx.reshape(-1)
	yy = np.zeros(coomat._shape[0])
	yy[:] = fill
	coo_matvec(coomat, xx, yy)
	return yy

def grid1_to_grid2(overlap) :
	tret = _glint2.grid1_to_grid2(_coo_to_tuple(overlap))
	return _tuple_to_coo(tret)

def grid2_to_grid1(overlap) :
	tret = _glint2.grid2_to_grid1(_coo_to_tuple(overlap))
	return _tuple_to_coo(tret)

def mask_out(overlap, mask1, mask2) :
	if mask1 is not None : mask1 = mask1.reshape(-1)
	if mask2 is not None : mask2 = mask2.reshape(-1)
	tret = _glint2.mask_out(_coo_to_tuple(overlap), mask1, mask2)
	return _tuple_to_coo(tret)

proj_native_area_correct = _glint2.proj_native_area_correct

def multiply_bydiag(a1, a2) :
#	print type(a1)
#	print type(a2)
	if issubclass(type(a1), scipy.sparse.coo_matrix) :
		a1 = _coo_to_tuple(a1)
	else :
		a2 = _coo_to_tuple(a2)
#	print a1
	return _glint2.multiply_bydiag(a1, a2)
