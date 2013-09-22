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

import giss.ncutil

# Stuff to read Snowdrift-format overlap, grid, etc. files

# ----------------------------------------------------------------
# Read a pair of (proj, llproj) out of a netCDF Overlap file
# @param var_name (eg: grid1)
def read_projs(nc, var_name) :
	info_name = var_name + '.info'
	sproj = str(nc.variables[info_name].projection)
	return giss.proj.make_projs(sproj)

# ----------------------------------------------------------------
def read_sparse_matrix(nc, var_name) :
	descr_var = nc.variables[var_name + '.descr']
	index_base = descr_var.index_base
	index = nc[var_name][:]
	index -= index_base
	row = index[:,0]
	col = index[:,1]
	val = nc[var_name + '.val'][:]

	return scipy.sparse.coo_matrix((val, (row, col)), shape=(descr_var.nrow, descr_var.ncol))

