#include <cstdio>
#include <map>

enum class MatrixStructure {GENERAL, SYMMETRIC, HERMETIAN, TRIANGULAR, ANTI_SYMMETRIC, DIAGONAL};
enum class TriangularType {GENERAL, LOWER, UPPER};
enum class MainDiagonalType {NON_UNIT, UNIT};

struct SparseBuilder {
	int _nrow, _ncol;

	// From the DESCRA array in NIST Sparse BLAS proposal
	MatrixStructure matrix_structure;
	TriangularType triangular_type;	
	MainDiagonalType main_diagonal_type;
	int array_base;

	/** NOTE: default operator<() for std::pair will produce row-major
	ordering of sparse elements, which is USUALLY what one wants. */
	std::map<std::pair<int, int>, double> _cells;

	// Used to map incoming indices (labels) to row and column numbers
	std::map<int,int> index2row;		// <incoming index, row #>
	std::map<int,int> index2col;
};


	extern "C"
	SparseBuilder *sparsebuilder_new_0(
		int const nrow, int const ncol, int array_base,
		MatrixStructure matrix_structure,
		TriangularType triangular_type,
		MainDiagonalType main_diagonal_type)
	{
		SparseBuilder *A = new SparseBuilder;
		A->array_base = array_base;
		A->_nrow = nrow;
		A->_ncol = ncol;

		A->matrix_structure = matrix_structure;
		A->triangular_type = triangular_type;
		A->main_diagonal_type = main_diagonal_type;

		return A;
	}


	extern "C"
	/** @param row_indices Array of row indices participating in this matrix,
		must have length(nrow) */
	SparseBuilder *sparsebuilder_setindices(
		SparseBuilder *A,
		int *row_indices,
		int *col_indices)
	{
		A->index2row.clear();
		for (int i=0; i<A->_nrow; ++i) {
//printf("index2row: %d->%d %d\n", row_indices[i], i+A->array_base, A->array_base);
			A->index2row.insert(std::make_pair(row_indices[i], i+A->array_base));
		}

		A->index2col.clear();
		for (int i=0; i<A->_ncol; ++i)
			A->index2col.insert(std::make_pair(col_indices[i], i+A->array_base));
	}


	extern "C"
	void sparsebuilder_delete_0(SparseBuilder *A)
	{
		delete A;
	}

	extern "C"
	int sparsebuilder_nrow(SparseBuilder *A) { return A->_nrow; }
	extern "C"
	int sparsebuilder_ncol(SparseBuilder *A) { return A->_ncol; }


	/** Replaces a value in a cell */
	extern "C"
	void sparsebuilder_set(SparseBuilder *A,
		int const _row, int const _col, double const val)
	{
		int row = _row;
		int col = _col;

		// Fix row and col for possible triangular type
		switch(A->triangular_type) {
			case TriangularType::UPPER :
				if (row > col) std::swap(row, col);
			break;
			case TriangularType::LOWER :
				if (row < col) std::swap(row, col);
			break;
		}


		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = A->_cells.find(std::make_pair(row, col));
		if (ii != A->_cells.end()) {
			ii->second = val;
		} else {
			A->_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}


	extern "C"
	bool sparsebuilder_set_byindex(SparseBuilder *A,
		int const irow, int const icol, double const val)
	{
		auto frow = A->index2row.find(irow);
		if (frow == A->index2row.end()) return false;
		int row = frow->second;

		auto fcol = A->index2col.find(icol);
		if (fcol == A->index2col.end()) return false;
		int col = fcol->second;

//printf("byindex: (%d->%d) (%d->%d)\n", irow, row, icol, col);

		sparsebuilder_set(A, row, col, val);
		return true;
	}


	/** Adds to the value in a cell */
	extern "C"
	void sparsebuilder_add(SparseBuilder *A,
		int const _row, int const _col, double const val)
	{
		int row = _row;
		int col = _col;

		// Fix row and col for possible triangular type
		switch(A->triangular_type) {
			case TriangularType::UPPER :
				if (row > col) std::swap(row, col);
			break;
			case TriangularType::LOWER :
				if (row < col) std::swap(row, col);
			break;
		}

		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = A->_cells.find(std::make_pair(row, col));
		if (ii != A->_cells.end()) {
			ii->second += val;
		} else {
			A->_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}



	/** Adds to the value in a cell */
	extern "C"
	bool sparsebuilder_add_byindex(SparseBuilder *A,
		int const irow, int const icol, double const val)
	{
		auto frow = A->index2row.find(irow);
		if (frow == A->index2row.end()) return false;
		int row = frow->second;

		auto fcol = A->index2col.find(icol);
		if (fcol == A->index2col.end()) return false;
		int col = fcol->second;

//printf("byindex: (%d->%d) (%d->%d)\n", irow, row, icol, col);

		sparsebuilder_add(A, row, col, val);
		return true;
	}



	/** @return NNZ (NIST Sparse BLAS), the number of point entries of the matrix.
	This is used to allocate arrays for rendering. */
	extern "C"
	int sparsebuilder_nnz(SparseBuilder *A) { return A->_cells.size(); }



	// For different sparse matrix formats, see:
    // http://math.nist.gov/~KRemington/fspblas/
	// (NIST Fortran Sparse BLAS: Toolkit Implementation)
    // A Revised Proposal for a Sparse BLAS Toolkit
	// (from Penn State's citeseer)
    // 	   csr - compressed sparse row
    // 	   csc - compressed sparse column
    // 	   coo - coordinate (matrix multiply only)
    // 	   bsr - block sparse row
    // 	   bsc - block sparse column
    // 	   bco - block coordinate (matrix multiply only)
    // 	   vbr - variable block row
    // 	   dia - diagonal
    // 	   ell - Ellpack
    // 	   jad - jagged diagonal
    // 	   sky - skyline
    // 	   bdi - block diagonal
    // 	   bel - block Ellpack 

	extern "C"
	void sparsebuilder_render_coo_0(SparseBuilder *A,
		double *val, int *indx, int *jndx)
	{
		int i = 0;
		for (auto ii = A->_cells.begin(); ii != A->_cells.end(); ++ii, ++i) {
			const int row = ii->first.first;
			const int col = ii->first.second;
			const double cell_val = ii->second;

			indx[i] = row;
			jndx[i] = col;
			val[i] = cell_val;
		}
	}

	extern "C"
	void sparsebuilder_sum_per_row_0(SparseBuilder *A, double *sum)
	{
		for (int i=0; i<A->_nrow; ++i) sum[i] = 0;

		for (auto ii = A->_cells.begin(); ii != A->_cells.end(); ++ii) {
			const int row_base0 = ii->first.first - A->array_base;
			const double cell_val = ii->second;

			sum[row_base0] += cell_val;
		}
	}

	extern "C"
	void sparsebuilder_sum_per_col_0(SparseBuilder *A, double *sum)
	{
		for (int i=0; i<A->_ncol; ++i) sum[i] = 0;

		for (auto ii = A->_cells.begin(); ii != A->_cells.end(); ++ii) {
			const int col_base0 = ii->first.second - A->array_base;
			const double cell_val = ii->second;

			sum[col_base0] += cell_val;
		}
	}

