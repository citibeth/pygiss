import scipy.sparse

def read_overlap(nc, vname) :
	"""Reads the overlap matrix from the Exchange Grid file"""

	cells_ijk = nc.variables[vname + '.cells.ijk'][:]
	rows = cells_ijk[:,0]
	cols = cells_ijk[:,1]

	info_var = nc.variables[vname + '.info']
	cells_area = nc.variables[vname + '.cells.area'][:]

	shape = (info_var.__dict__['grid1.ncells_full'],
		info_var.__dict__['grid2.ncells_full'])
	return scipy.sparse.coo_matrix((cells_area, (rows, cols)), shape=shape)
