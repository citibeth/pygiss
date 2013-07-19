import scipy.sparse
import numpy as np

# Complement to C++ SparseMatrix::netcdf_define()
def read_coo_matrix(nc, vname) :
	vv = nc.variables[vname + '.index']
	cells_ijk = np.zeros(vv.shape, dtype='i')
	cells_ijk[:] = vv[:]
	rows = cells_ijk[:,0]
	cols = cells_ijk[:,1]
	vals = nc.variables[vname + '.val'][:]

	descr_var = nc.variables[vname + '.descr']

	shape = (descr_var.nrow, descr_var.ncol)
	return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=shape)
