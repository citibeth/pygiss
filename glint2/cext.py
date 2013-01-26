import _glint2
import scipy.sparse

# Interface with C++ extension

def grid1_to_grid2(overlap) :
	print type(overlap.row)
	print type(overlap.col)
	print type(overlap.data)


	(nrow1, ncol1, rows1, cols1, data1) = _glint2.grid1_to_grid2((
		overlap._shape[0], overlap._shape[1],
		overlap.row, overlap.col, overlap.data))

	print ncol1, nrow1
	print type(rows1)
	print type(cols1)
	print type(data1)

	return scipy.sparse.coo_matrix((data1, (rows1, cols1)), shape=(nrow1, ncol1))

