#pragma once

namespace giss {


class SparseMatrixBuilder {
	enum { DUP_ADD, DUP_REPLACE } DupBehavior;
};

enum class {GENERAL, SYMMETRIC, HERMETIAN, TRIANGULAR, ANTI_SYMMETRIC, DIAGONAL} MatrixStructure;
enum class {GENERAL, LOWER, UPPER} TriangularType;
enum class {NON_UNIT, UNIT} MainDiagonalType;

struct SparseMatrixBuilder {

	int _nrow, _ncol;

	/** NOTE: default operator<() for std::pair will produce row-major
	ordering of sparse elements, which is USUALLY what one wants. */
	std::map<std::pair<int, int>, double> _cells;

	// From the DESCRA array in NIST Sparse BLAS proposal
	MatrixStructure matrix_structure;
	TriangularType triangular_type;	
	MainDiagonalType main_diagonal_type;


	std::map<int, int> index2row;
	std::map<int, int> index2col;
};

	SparseMatrixBuilder *SparseMatrixBuilder_new(SparseMatrixBuilder *A,
		int nrow, int ncol, DupBehavior dup_behavior)
	{
		A->_nrow = nrow;
		A->_ncol = ncol;
		A->_dup_behavior =dup_behavior;
	}

	void SparseMatrixBuilder_nrow(SparseMatrixBuilder *A) { return A->_nrow; }
	void SparseMatrixBuilder_ncol(SparseMatrixBuilder *A) { return A->_ncol; }


	/** Replaces a value in a cell */
	void SparseMatrixBuilder_set(int row, int col, double val)
	{
		// Fix row and col for possible triangular type
		switch(triangular_type) {
			case TriangularType::UPPER :
				if (row >= col) std::swap(row, col);
			break;
			case TriangularType::LOWER :
				if (row <= col) std::swap(row, col);
			break;
		}


		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = _cells.find(std::make_pair(row, col));
		if (ii != _cells.end()) {
			ii->second = val;
		} else {
			_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}

	/** Adds to the value in a cell */
	void SparseMatrixBuilder_add(int row, int col, double val)
	{
		// Fix row and col for possible triangular type
		switch(triangular_type) {
			case TriangularType::UPPER :
				if (row >= col) std::swap(row, col);
			break;
			case TriangularType::LOWER :
				if (row <= col) std::swap(row, col);
			break;
		}

		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = _cells.find(std::make_pair(row, col));
		if (ii != _cells.end()) {
			ii->second += val;
		} else {
			_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}


	/** @return NNZ (NIST Sparse BLAS), the number of point entries of the matrix.
	This is used to allocate arrays for rendering. */
	int SparseMatrixBuilder_NNZ() { return _cells.size(); }



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

	void SparseMatrixBuilder_render_coo(double *descra, double *val, int *indx, int *jndx)
	{
		descra -= 1;	// Convert to FORTRAN indexing
		descra[1] = (int)matrix_structure;
		descra[2] = (int)triangular_type;
		descra[3] = (int)main_diagonal_type;
		descra[4] = array_base;
		descra[6] = 1;		// No repeated values in columns of indx

		int i = 0;
		for (auto ii = _cells.begin(); ii != _cells.end(); ++ii, ++i) {
			int row = ii->first.first;
			int col = ii->first.second;
			double cell_val = ii->second;

			indx[i] = row;
			jndx[i] = col;
			val[i] = cell_val;
		}
	}




}



#if 0
	class Cell {
		int row;
		int col : 31;
		bool replace : 1;	// Replace previous values with this?  (Or add)?
		double val;

		Cell(int _row, int _col, double _val, bool _replace) :
			row(_row), col(_col), val(_val), replace(_replace) {}
	};
#endif
