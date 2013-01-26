import _glint2
import scipy.sparse
import numpy as np

# Interface with C++ extension

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


# Puts A*x into y, does not overwrite unused elements of y
# @param yy OUTPUT
def coo_matvec(coomat, xx, yy) :
	_glint2.coo_matvec(_coo_to_tuple(coomat), xx, yy)
	return

def coo_multiply(coomat, xx, fill=np.nan) :
	yy = np.zeros(coomat._shape[0])
	yy[:] = fill
	coo_matvec(coomat, xx, yy)
	return yy

def grid1_to_grid2(overlap) :
	tret = _glint2.grid1_to_grid2(_coo_to_tuple(overlap))
	return _tuple_to_coo(tret)

