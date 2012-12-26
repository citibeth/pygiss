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

