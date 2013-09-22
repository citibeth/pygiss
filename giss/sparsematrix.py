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
