namespace giss {

// ------------------------------------------------------------
enum class MatrixStructure {GENERAL, SYMMETRIC, HERMETIAN, TRIANGULAR, ANTI_SYMMETRIC, DIAGONAL};
enum class TriangularType {GENERAL, LOWER, UPPER};
enum class MainDiagonalType {NON_UNIT, UNIT};
//enum class DuplicatePolicy {UNDEFINED, REPLACE, ADD, BOTH};

// ------------------------------------------------------------

// Matches up with Fortran's sparsecoord_t
class SparseDescr {
public:
	virtual ~SparseDescr() {}

	// From the DESCRA array in NIST Sparse BLAS proposal
	MatrixStructure matrix_structure;
	TriangularType triangular_type;	
	MainDiagonalType main_diagonal_type;
	int index_base;

	// DuplicatePolicy const duplicate_policy;

	int const nrow;
	int const ncol;
	// int const nnz;

	SparseDescr(
		int const _nrow, int const _ncol, int index_base = 0,
		MatrixStructure matrix_structure = MatrixStructure::GENERAL,
		TriangularType triangular_type = TriangularType::GENERAL,
		MainDiagonalType main_diagonal_type = MainDiagonalType::NON_UNIT) :
//		DuplicatePolicy _duplicate_policy) :

		nrow(_nrow),
		ncol(_ncol),
		index_base(_index_base),
		matrix_structure(_matrix_structure),
		triangular_type(_triangular_type),
		main_diagonal_type(_main_diagonal_type)
//		duplicate_policy(_duplicate_policy)
	{}


	void _set(int row, int col, double const val) {
		fprintf("set() operation not implemented\n");
		throw std::exeption();
	}
	void _add(int row, int col, double const val) {
		fprintf("add() operation not implemented\n");
		throw std::exeption();
	}

	virtual size_t size();

private:
	void netcdf_write(NcFile *nc, std::string const &vname)
	{
		NcVar *grid_indexVar = nc->get_var((vname + ".num_elements").c_str());
		NcVar *areaVar = nc->get_var((vname + ".val").c_str());

		int i=0;
		for (auto ov = begin(); ov != end(); ++ov) {
			grid_indexVar->set_cur(i,0);
			int index[2] = {ov.row() + index_base, ov.col() + index_base};
			grid_indexVar->put(index, 1,2);

			areaVar->set_cur(i);
			areaVar->put(&ov->val(), 1);

			++i;
		}
	}

public:
	boost::function<void ()> netcdf_define(std::string const &vname, NcFile &nc)
	{
		auto lenDim = nc.add_dim((vname + ".num_elements").c_str(), size());
		auto num_gridsDim = nc.add_dim((vname + ".rank").c_str(), 2);
		auto grid_indexVar = nc.add_var((vname + ".index").c_str(), ncInt, lenDim, num_gridsDim);
		auto areaVar = nc.add_var((vname + ".val").c_str(), ncDouble, lenDim);

		return boost::bind(&SparseDescr::netcdf_write, this, &nc, vname);
	}



};
// =================================================================
// Three different kinds of Sparse matrices

// ------------------------------------------------------------
/** Matrix based on external Fortran HSL_ZD11 storage */
class ZD11SparseMatrix : public SparseDescr
{
protected:
	// Current number of elements in matrix.  Matrix is not
	// valid until this equals zd11.ne
	int _nnz_cur;

public:

	// --------------------------------------------------
	class iterator {
	protected:
		ZD11 *zd11;
		int i;
		iterator(ZD11 *z, int _i) : zd11(z), i(_i) {}
	public:
		operator==(iterator const &rhs) { return i == rhs.i; }
		operator!=(iterator const &rhs) { return i != rhs.i; }
		operator++(iterator const &4hs) { ++i; }
		int &row() { return zd11->row[i] - zd11->index_base; }
		int &col() { return zd11->col[i] - zd11->index_base; }
		double &val() { return zd11->val[i]; }
	};
	iterator begin() { return iterator(0); }
	iterator end() { return iterator(_nnz_cur); }
	// --------------------------------------------------


	// Pointers/references to main storage
	ZD11 &zd11;


	/** Call this after ZD11 has been initialized
	@param _zd11 Pointer to Fortran structure. */
	ZD11SparseMatrix(ZD11 &_zd11, int nnz_cur,
		MatrixStructure matrix_structure,
		TriangularType triangular_type,
		MainDiagonalType main_diagonal_type)
	: SparseDescr(_zd11.m, _zdll.n, 1,
	matrix_structure, triangular_type, main_diagonal_type),
	_nnz_cur(nnz_cur), zd11(_zd11)
	{}

	void clear() { _nnz_cur = 0; }

	bool is_complete() { return _nnz_cur == zd11.ne; }

	size_t size() { return _nnz_cur; }

private:
	void _set(int const row, int const col, double const val)
	{
		if (_nnz_cur >= zd11.ne) {
			fprintf(stderr, "ZD11SparseMatrix is full with %d elements\n", zd11.ne);
			throw std::exception();
		}
		zd11.row[_nnz_cur] = row;
		zd11.col[_nnz_cur] = col;
		zd11.val[_nnz_cur] = val;
		++_nnz_cur;
	}

#include "SparseMatrixMethods.cpp.hpp"
};
// ---------------------------------------------------------
/** SparseMatrix as read out of a netCDF file in 3 parallel std::vector<> arrays */
class VectorSparseMatrix : public SparseDescr
{
protected:
	std::vector<int> indx;
	std::vector<int> jndx;
	std::vector<double> val;
public:

	// --------------------------------------------------
	class iterator {
	protected:
		VectorSparseMatrix *parent;
		int i;
		iterator(VectorSparseMatrix *p, int _i) : parent(p), i(_i) {}
	public:
		operator==(iterator const &rhs) { return i == rhs.i; }
		operator!=(iterator const &rhs) { return i != rhs.i; }
		operator++(iterator const &4hs) { ++i; }
		int &row() { return parent->indx[i] - parent->index_base; }
		int &col() { return parent->jndx[i] - parent->index_base; }
		double &val() { return parent->val[i]; }
	};
	iterator begin() { return iterator(this, 0); }
	iterator end() { return iterator(this, val.size()); }
	// --------------------------------------------------

	void clear() {
		indx.clear();
		jndx.clear();
		val.clear();
	}
	size_t size() { return val.size(); }

	/** Construct from existing vectors */
	VectorSparseMatrix(SparseDescr const &descr,
		std::vector<int> &&_indx,
		std::vector<int> &&_jndx,
		std::vector<double> &&_val) :
	: SparseDescr(descr),
	indx(std::move(_indx)), jndx(std::move(_jndx)), val(std::move(_val))
	{}

	/** Construct a new one */
	VectorSparseMatrix(SparseDescr const &descr) : SparseDescr(descr) {}

	void sort_row_major() {...}

protected :
	void _set(int row, int col, double _val)
	{
		indx.push_back(row);
		jndx.push_back(col);
		val.push_back(_val);
	}

#include "SparseMatrixMethods.cpp.hpp"
}
// ----------------------------------------------------------
class MapSparseMatrix : public SparseDescr {
protected :
	std::map<std::pair<int,int>, double> _cells;
	typedef std::map<std::pair<int,int>, double>::iterator ParentIterator;
public:

	// --------------------------------------------------
	class iterator : public ParentIterator {
	public:
		MapSparseMatrix *parent;

		int &row() { return first.first - parent->index_base; }
		int &col() { return first.second - parent->index_base; }
		double &val() { return second; }

		iterator(ParentIterator const &_i, MapSparseMatrix *p) : ParentIterator(_i), parent(p) {}
	};
	iterator begin() { return _cells.begin(); }
	iterator end() { return _cells.end(); }
	// --------------------------------------------------

	MapSparseMatrix(SparseDescr const &descr) :
		SparseDescr(descr) {}

	void clear() { _cells.clear(); }

	size_t size() { return _cells.size(); }

	void _set(int row, int col, double const val)
	{
		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = A->_cells.find(std::make_pair(row, col));
		if (ii != A->_cells.end()) {
			ii->second = val;
		} else {
			A->_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}

	void _add(int row, int col, double const val)
	{
		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = A->_cells.find(std::make_pair(row, col));
		if (ii != A->_cells.end()) {
			ii->second += val;
		} else {
			A->_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}

#include "SparseMatrixMethods.cpp.hpp"
};
// ============================================================
// Helper Classes for building matrices with dictionaries

// ------------------------------------------------------------
/** Helper in creating matrices */
template<class IndexT>
class IndexMap {
	std::map<IndexT, int> _map;
public:

	IndexMap(std::set<IndexT> const &indices)
	{
		int i=0;
		for (auto ii = indices.begin(); ii != indices.end(); ++ii) {
			_map.push_back(std::make_pair(*ii, i++));
		}
	}

	int operator()(IndexT const &ix) const {
		auto ii _map.find(ix);
		if (ii == _map.end()) {
			fprintf(stderr, "Index not in map: %d\n", ix);
			throw std::exception();
		}
		return ii->second;
	}
};


template<class SparseMatrixT, class IndexT>
class IndexedMatrixBuilder
{
public :
	IndexMap<IndexT> const rowi;
	IndexMap<IndexT> const coli;
	SparseMatrixT * const matrix;
	
	IndexedMatrixBuilder(SparseMatrixT *sm,
		std::set<IndexT> const &rows,
		std::set<IndexT> const &cols) :
		matrix(sm), rowi(rows), coli(cols)
	{}

	void set(int const _row, int const _col, double const val)
		{ sm->set(rowi(_row), coli(_col), val); }
	void add(int const _row, int const _col, double const val) = 0;
		{ sm->add(rowi(_row), coli(_col), val); }
};
// ------------------------------------------------------------
/** Copy a to b.  Does not clear b */
template<SparseMatrixT1, SparseMatrixT2>
void copy_set(SparseMatrixT1 &a, SparseMatrixT2 &b)
{
	for (SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		b.set(ii.row(), ii.col(), ii.val());
	}
}

/** Copy a to b.  Does not clear b */
template<SparseMatrixT1, SparseMatrixT2>
void copy_add(SparseMatrixT1 &a, SparseMatrixT2 &b)
{
	for (SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		b.add(ii.row(), ii.col(), ii.val());
	}
}

// ------------------------------------------------------------

/// Computes y = A * x
template<SparseMatrixT>
void multiply(SparseMatrixT &A, double const * x, double *y)
{
	int nx = A.ncol;
	int ny = A.nrow;
	for (int iy = 0; iy < ny; ++iy) y[iy] = 0;
	for (SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		int ix = ii.col();
		int iy = ii.row();
		y[iy] += ii.val() * x[ix];
	}
}







template<SparseMatrixT>
std::vector<double> sum_per_row(SparseMatrixT &M) {
	std::vector<double> ret(nrow);
	for (SparseMatrixT::iterator ii = begin(); ii != end(); ++ii) {
		ret[ii.row()] += ii.val();
	}
	return ret;
}

template<SparseMatrixT>
std::vector<double> sum_per_col(SparseMatrixT &M) {
	std::vector<double> ret(ncol);
	for (SparseMatrixT::iterator ii = begin(); ii != end(); ++ii) {
		ret[ii.col()] += ii.val();
	}
	return ret;
}


template<SparseMatrixT>
std::map<int,double> sum_per_row_map(SparseMatrixT &M) {
	std::map<int,double> ret;
	for (SparseMatrixT::iterator ii = begin(); ii != end(); ++ii) {
		auto f = ret.find(ii.row());
		if (f == ret.end()) {
			ret.insert(std::make_pair(ii.row(), ii.val());
		} else {
			f->second += ii.val();
		}
	}
	return ret;
}

template<SparseMatrixT>
std::map<int,double> sum_per_col_map(SparseMatrixT &M) {
	std::map<int,double> ret;
	for (SparseMatrixT::iterator ii = begin(); ii != end(); ++ii) {
		auto f = ret.find(ii.col());
		if (f == ret.end()) {
			ret.insert(std::make_pair(ii.col(), ii.val());
		} else {
			f->second += ii.val();
		}
	}
	return ret;
}













// ------------------------------------------------------------
// /** Describes a cell in a sparse matrix */
// struct SparseCell {
// 	int index[2];	// The cell index of gridcell in each grid that overlaps
// 	double val;	// Area of overlap
// 
// 	SparseCell(int _index0, int _index1, double _val) :
// 		index({_index0, _index1}), val(_val) {}
// 
// 	/** Sort overlap matrix in row major form, common for sparse matrix representations */
// 	bool operator<(SparseCell const &b) const
// 	{
// 		if (index[0] < b.index[0]) return true;
// 		if (index[0] > b.index[0]) return false;
// 		return (index[1] < b.index[1]);
// 	}
// };
// 
// 
// class VectorSparseMatrix : public SparseMatrix {
// 	std::vector<SparseCell> _cells;
// 
// public:
// 	SparseCell *cells()
// 	{ return &_cells[0]; }
// 
// 	VectorSparseMatrix(SparseMatrix const &descr) :
// 		this(descr) {}
// 
// 	int nnz() { return _cells.size(); }
// 
// 	void set(int const _row, int const _col, double const _val)
// 		{ _cells.push_back(SparseCell(_row, _col, _val)); }
// 
// 	std::vector<double> sum_per_row() {
// 		std::vector<double> ret(nrow);
// 		for (auto ii = _cells.begin(); ii != _cells.end(); ++ii) {
// 			ret[ii->index[0]] += ii->val;
// 		}
// 		return ret;
// 	}
// 
// 	std::vector<double> sum_per_col() {
// 		std::vector<double> ret(ncol);
// 		for (auto ii = _cells.begin(); ii != _cells.end(); ++ii) {
// 			ret[ii->index[1]] += ii->val;
// 		}
// 		return ret;
// 	}
// };
// 
// VectorSparseMatrix_cells_f(VectorSparseMatrix *M)
// 	{ return &M->cells[0]; }





}		// namespace giss
