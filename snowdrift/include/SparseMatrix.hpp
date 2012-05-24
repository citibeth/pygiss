#pragma once 

#include <map>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include "hsl_zd11_x.hpp"
#include "ncutil.hpp"

class NcFile;

namespace giss {

// ------------------------------------------------------------

// ------------------------------------------------------------


// Matches up with Fortran's sparsecoord_t
class SparseDescr {
public:

enum class MatrixStructure {GENERAL, SYMMETRIC, HERMETIAN, TRIANGULAR, ANTI_SYMMETRIC, DIAGONAL};
enum class TriangularType {GENERAL, LOWER, UPPER};
enum class MainDiagonalType {NON_UNIT, UNIT};
enum class DuplicatePolicy {REPLACE, ADD};
enum class SortOrder {ROW_MAJOR, COLUMN_MAJOR};

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
		int const _nrow, int const _ncol, int _index_base = 0,
		MatrixStructure _matrix_structure = MatrixStructure::GENERAL,
		TriangularType _triangular_type = TriangularType::GENERAL,
		MainDiagonalType _main_diagonal_type = MainDiagonalType::NON_UNIT) :

		nrow(_nrow),
		ncol(_ncol),
		index_base(_index_base),
		matrix_structure(_matrix_structure),
		triangular_type(_triangular_type),
		main_diagonal_type(_main_diagonal_type)
	{}
};

class SparseMatrix : public SparseDescr {
public:
	SparseMatrix(SparseDescr const &descr) : SparseDescr(descr) {}

	virtual ~SparseMatrix() {}
	virtual size_t size() = 0;
	virtual void set(int row, int col, double const val, DuplicatePolicy dups = DuplicatePolicy::REPLACE) = 0;
	void add(int row, int col, double const val)
		{ set(row, col, val, DuplicatePolicy::ADD); }

	virtual boost::function<void ()> netcdf_define(NcFile &nc, std::string const &vname) = 0;

	virtual void multiply(double const * x, double *y, bool clear_y = true) = 0;
	virtual void multiplyT(double const * x, double *y, bool clear_y = true) = 0;
	virtual std::vector<double> sum_per_row() = 0;
	virtual std::vector<double> sum_per_col() = 0;
	virtual std::map<int,double> sum_per_row_map() = 0;
	virtual std::map<int,double> sum_per_col_map() = 0;

};

// =================================================================
// Mix-ins common to all sparse matrix types

template<class SparseMatrix0T>
class SparseMatrix1 : public SparseMatrix0T
{
protected:
	SparseMatrix1(SparseDescr const &descr) :
		SparseMatrix0T(descr) {}
public:
	void set(int row, int col, double const val, SparseMatrix::DuplicatePolicy dups = SparseMatrix::DuplicatePolicy::REPLACE);
	boost::function<void ()> netcdf_define(NcFile &nc, std::string const &vname);

	void multiply(double const * x, double *y, bool clear_y = true);
	void multiplyT(double const * x, double *y, bool clear_y = true);
	std::vector<double> sum_per_row();
	std::vector<double> sum_per_col();
	std::map<int,double> sum_per_row_map();
	std::map<int,double> sum_per_col_map();

private:
	void netcdf_write(NcFile *nc, std::string const &vname);
};


template<class SparseMatrix0T>
void SparseMatrix1<SparseMatrix0T>::set(int row, int col, double const val, SparseMatrix::DuplicatePolicy dups)
{
	// Fix row and col for possible triangular type
	switch(this->triangular_type) {
		case SparseMatrix::TriangularType::UPPER :
			if (row > col) std::swap(row, col);
		break;
		case SparseMatrix::TriangularType::LOWER :
			if (row < col) std::swap(row, col);
		break;
	}

	// Check range
	if (row >= this->nrow || row < 0) {
		fprintf(stderr, "SparseMatrix1<>::set(), row=%d >= nrow=%d or <0\n", row, this->nrow);
		throw std::exception();
	}
	if (col >= this->ncol || col < 0) {
		fprintf(stderr, "SparseMatrix1<>::set(), col=%d >= ncol=%d or <0\n", col, this->ncol);
		throw std::exception();
	}

	// Adjust for index_base
	row += this->index_base;
	col += this->index_base;

	this->_set(row, col, val, dups);
}


template<class SparseMatrix0T>
void SparseMatrix1<SparseMatrix0T>::netcdf_write(NcFile *nc, std::string const &vname)
{
	NcVar *grid_indexVar = nc->get_var((vname + ".index").c_str());
	NcVar *areaVar = nc->get_var((vname + ".val").c_str());

	int i=0;
	for (typename SparseMatrix1<SparseMatrix0T>::iterator ov = this->begin(); ov != this->end(); ++ov) {
		grid_indexVar->set_cur(i,0);
		int index[2] = {ov.row() + this->index_base, ov.col() + this->index_base};
		grid_indexVar->put(index, 1,2);

		areaVar->set_cur(i);
		areaVar->put(&ov.val(), 1);

		++i;
	}
}


template<class SparseMatrix0T>
boost::function<void ()> SparseMatrix1<SparseMatrix0T>::netcdf_define(
	NcFile &nc, std::string const &vname)
{
	auto lenDim = nc.add_dim((vname + ".num_elements").c_str(), this->size());
	auto num_gridsDim = nc.add_dim((vname + ".rank").c_str(), 2);
	auto grid_indexVar = nc.add_var((vname + ".index").c_str(), ncInt, lenDim, num_gridsDim);
	auto areaVar = nc.add_var((vname + ".val").c_str(), ncDouble, lenDim);

	auto oneDim = get_or_add_dim(nc, "one", 1);
	auto descrVar = nc.add_var((vname + ".descr").c_str(), ncInt, oneDim);	// TODO: This should be ".info"
	descrVar->add_att("nrow", this->nrow);
	descrVar->add_att("ncol", this->ncol);
	descrVar->add_att("index_base", this->index_base);
	descrVar->add_att("matrix_structure", (int)this->matrix_structure);
	descrVar->add_att("triangular_type", (int)this->triangular_type);
	descrVar->add_att("main_diagonal_type", (int)this->main_diagonal_type);

	return boost::bind(&SparseMatrix1<SparseMatrix0T>::netcdf_write,
		this, &nc, vname);
}

/// Computes y = A * x
template<class SparseMatrix0T>
void SparseMatrix1<SparseMatrix0T>::multiply(double const * x, double *y, bool clear_y)
{
	int nx = this->ncol;
	int ny = this->nrow;
	if (clear_y) for (int iy = 0; iy < ny; ++iy) y[iy] = 0;
	for (auto ii = this->begin(); ii != this->end(); ++ii) {
		int ix = ii.col();
		int iy = ii.row();
		y[iy] += ii.val() * x[ix];
	}
}

/// Computes y = A^T * x
template<class SparseMatrix0T>
void SparseMatrix1<SparseMatrix0T>::multiplyT(double const * x, double *y, bool clear_y)
{
	int nx = this->nrow;
	int ny = this->ncol;
	if (clear_y) for (int iy = 0; iy < ny; ++iy) y[iy] = 0;
	for (auto ii = this->begin(); ii != this->end(); ++ii) {
		int iy = ii.col();
		int ix = ii.row();
		y[iy] += ii.val() * x[ix];
	}
}



// ------------------------------------------------------------
template<class SparseMatrix0T>
std::vector<double> SparseMatrix1<SparseMatrix0T>::sum_per_row() {
	std::vector<double> ret(this->nrow);
	for (auto ii = this->begin(); ii != this->end(); ++ii) {
		ret[ii.row()] += ii.val();
	}
	return ret;
}

template<class SparseMatrix0T>
std::vector<double> SparseMatrix1<SparseMatrix0T>::sum_per_col() {
	std::vector<double> ret(this->ncol);
	for (auto ii = this->begin(); ii != this->end(); ++ii) {
		ret[ii.col()] += ii.val();
	}
	return ret;
}


template<class SparseMatrix0T>
std::map<int,double> SparseMatrix1<SparseMatrix0T>::sum_per_row_map() {
	std::map<int,double> ret;
	for (auto ii = this->begin(); ii != this->end(); ++ii) {
		auto f = ret.find(ii.row());
		if (f == ret.end()) {
			ret.insert(std::make_pair(ii.row(), ii.val()));
		} else {
			f->second += ii.val();
		}
	}
	return ret;
}

template<class SparseMatrix0T>
std::map<int,double> SparseMatrix1<SparseMatrix0T>::sum_per_col_map() {
	std::map<int,double> ret;
	for (auto ii = this->begin(); ii != this->end(); ++ii) {
		auto f = ret.find(ii.col());
		if (f == ret.end()) {
			ret.insert(std::make_pair(ii.col(), ii.val()));
		} else {
			f->second += ii.val();
		}
	}
	return ret;
}

// ------------------------------------------------------------




// =================================================================
// Three different kinds of Sparse matrices

// ------------------------------------------------------------
/** Matrix based on external Fortran HSL_ZD11 storage */
class ZD11SparseMatrix0 : public SparseMatrix
{
protected:
	// Current number of elements in matrix.  Matrix is not
	// valid until this equals zd11.ne
	int _nnz_cur;

	// Pointers/references to main storage
	ZD11 *_zd11;

	ZD11SparseMatrix0(SparseDescr const &descr) : SparseMatrix(descr) {}
public:

	ZD11 &zd11() { return *_zd11; }

	// --------------------------------------------------
	class iterator {
	protected:
		ZD11SparseMatrix0 *parent;
		int i;
	public:
		iterator(ZD11SparseMatrix0 *z, int _i) : parent(z), i(_i) {}
		bool operator==(iterator const &rhs) { return i == rhs.i; }
		bool operator!=(iterator const &rhs) { return i != rhs.i; }
		void operator++() { ++i; }
		int row() { return parent->zd11().row[i] - parent->index_base; }
		int col() { return parent->zd11().col[i] - parent->index_base; }
		double &val() { return parent->zd11().val[i]; }
	};
	iterator begin() { return iterator(this, 0); }
	iterator end() { return iterator(this, _nnz_cur); }
	// --------------------------------------------------

	void clear() { _nnz_cur = 0; }

	bool is_complete() { return _nnz_cur == zd11().ne; }

	size_t size() { return _nnz_cur; }

protected:
	void _set(int const row, int const col, double const val, DuplicatePolicy dups)
	{
		if (_nnz_cur >= zd11().ne) {
			fprintf(stderr, "ZD11SparseMatrix is full with %d elements\n", zd11().ne);
			throw std::exception();
		}
		zd11().row[_nnz_cur] = row;
		zd11().col[_nnz_cur] = col;
		zd11().val[_nnz_cur] = val;
		++_nnz_cur;
	}
};
// -----------------------------------------------------------------
class ZD11SparseMatrix : public SparseMatrix1<ZD11SparseMatrix0>
{
public:
	/** Call this after ZD11 has been initialized
	@param _zd11 Pointer to Fortran structure. */
	ZD11SparseMatrix(ZD11 &__zd11, int nnz_cur,
		MatrixStructure matrix_structure = MatrixStructure::GENERAL,
		TriangularType triangular_type = TriangularType::GENERAL,
		MainDiagonalType main_diagonal_type = MainDiagonalType::NON_UNIT)
	: SparseMatrix1<ZD11SparseMatrix0>(SparseDescr(__zd11.m, __zd11.n, 1,
	matrix_structure, triangular_type, main_diagonal_type))
	{
		_nnz_cur = nnz_cur;
		_zd11 = &__zd11;
	}
};
// ==================================================================
// ---------------------------------------------------------
/** SparseMatrix as read out of a netCDF file in 3 parallel std::vector<> arrays */
class VectorSparseMatrix0 : public SparseMatrix
{
protected:
	std::vector<int> indx;
	std::vector<int> jndx;
	std::vector<double> val;

	VectorSparseMatrix0(SparseDescr const &descr) : SparseMatrix(descr) {}

public:

	// --------------------------------------------------
	class iterator {
	protected:
		VectorSparseMatrix0 *parent;
		int i;
	public:
		iterator(VectorSparseMatrix0 *p, int _i) : parent(p), i(_i) {}
		bool operator==(iterator const &rhs) { return i == rhs.i; }
		bool operator!=(iterator const &rhs) { return i != rhs.i; }
		void operator++() { ++i; }
		int row() { return parent->indx[i] - parent->index_base; }
		int col() { return parent->jndx[i] - parent->index_base; }
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

protected :
	void _set(int row, int col, double _val, DuplicatePolicy dups)
	{
		indx.push_back(row);
		jndx.push_back(col);
		val.push_back(_val);
	}
};

class VectorSparseMatrix : public SparseMatrix1<VectorSparseMatrix0>
{
public:
	VectorSparseMatrix(SparseDescr const &descr) :
	SparseMatrix1<VectorSparseMatrix0>(descr)
	{}


	/** Construct from existing vectors */
	VectorSparseMatrix(SparseDescr const &descr,
		std::vector<int> &&_indx,
		std::vector<int> &&_jndx,
		std::vector<double> &&_val) :
	SparseMatrix1<VectorSparseMatrix0>(descr)
	{
		indx = std::move(_indx);
		jndx = std::move(_jndx);
		val = std::move(_val);
	}

	void sort(SparseMatrix::SortOrder sort_order = SortOrder::ROW_MAJOR);

	static std::unique_ptr<VectorSparseMatrix> netcdf_read(NcFile &nc, std::string const &vname);

};
// ====================================================================
// ----------------------------------------------------------
class MapSparseMatrix0 : public SparseMatrix {
protected :
	std::map<std::pair<int,int>, double> _cells;
	typedef std::map<std::pair<int,int>, double>::iterator ParentIterator;
	MapSparseMatrix0(SparseDescr const &descr) : SparseMatrix(descr) {}

public:

	// --------------------------------------------------
	class iterator {
		ParentIterator ii;
		MapSparseMatrix0 *parent;
	public:

		iterator(ParentIterator const &_i, MapSparseMatrix0 *p) : ii(_i), parent(p) {}
		bool operator==(iterator const &rhs) { return ii == rhs.ii; }
		bool operator!=(iterator const &rhs) { return ii != rhs.ii; }
		void operator++() { ++ii; }
		int row() { return ii->first.first - parent->index_base; }
		int col() { return ii->first.second - parent->index_base; }
		double &val() { return ii->second; }
	};
	iterator begin() { return iterator(_cells.begin(), this); }
	iterator end() { return iterator(_cells.end(), this); }
	// --------------------------------------------------

	void clear() { _cells.clear(); }

	size_t size() { return _cells.size(); }

protected :
	void _set(int row, int col, double const val, DuplicatePolicy dups)
	{
		// Could make this find-insert operation a bit more efficient
		// by only indexing into the std::map once...
		auto ii = _cells.find(std::make_pair(row, col));
		if (ii != _cells.end()) {
			if (dups == DuplicatePolicy::ADD) ii->second += val;
			else ii->second = val;
		} else {
			_cells.insert(std::make_pair(std::make_pair(row, col), val));
		}
	}
};
// ---------------------------------------------------------------
class MapSparseMatrix : public SparseMatrix1<MapSparseMatrix0>
{
public:
	MapSparseMatrix(SparseDescr const &descr) :
		SparseMatrix1<MapSparseMatrix0>(descr) {}
};
// ============================================================


// ------------------------------------------------------------
/** Copy a to b.  Does not clear b */
template<class SparseMatrixT1, class SparseMatrixT2>
void copy(SparseMatrixT1 &a, SparseMatrixT2 &b, SparseMatrix::DuplicatePolicy dups = DuplicatePolicy::REPLACE)
{
	b.clear();
	for (typename SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		b.set(ii.row(), ii.col(), ii.val(), dups);
	}
}
// ------------------------------------------------------------
template<class SparseMatrixT1, class SparseMatrixT2>
inline void make_used_translators(SparseMatrixT1 &a,
IndexTranslator &trans_row,
IndexTranslator &trans_col,
std::set<int> *_used_row,	// If not NULL, output to here.
std::set<int> *_used_col)	// If not NULL, output to here.

{
	// Figure out what is used
	std::set<int> used_row;
	std::set<int> used_col;
	for (typename SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		used_row.insert(ii.row());
		used_col.insert(ii.col());
	}

	// Convert used sets to translators
	trans_row.init(a.nrow, used_row);
	trans_col.init(a.ncol, used_col);

	if (_used_row) *_used_row = std::move(used_row);
	if (_used_col) *_used_col = std::move(used_col);
}


template<class SparseMatrixT1>
inline void make_used_row_translator(SparseMatrixT1 &a,
IndexTranslator &trans_row,
std::set<int> *_used_row)	// If not NULL, output to here.
{
	// Figure out what is used
	std::set<int> used_row;
	for (typename SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		used_row.insert(ii.row());
	}

	// Convert used sets to translators
	trans_row.init(a.nrow, used_row);
	if (_used_row) *_used_row = std::move(used_row);
}

template<class SparseMatrixT1>
inline void make_used_col_translator(SparseMatrixT1 &a,
IndexTranslator &trans_col,
std::set<int> *_used_col)	// If not NULL, output to here.
{
	// Figure out what is used
	std::set<int> used_col;
	for (typename SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		used_col.insert(ii.col());
	}

	// Convert used sets to translators
	trans_col.init(a.ncol, used_col);
	if (_used_col) *_used_col = std::move(used_col);
}



template<class SparseMatrixT1, class SparseMatrixT2>
inline void translate_indices(SparseMatrixT1 &a, SparseMatrixT2 &b,
IndexTranslator const &trans_row,
IndexTranslator const &trans_col,
double *row_sum = NULL,		// Place to sum row and column values, if it exists
double *col_sum = NULL,
SparseMatrix::DuplicatePolicy dups = DuplicatePolicy::REPLACE)
{
	b.clear();
	if (row_sum) for (int i=0; i<b.nrow; ++i) row_sum[i] = 0;
	if (col_sum) for (int i=0; i<b.ncol; ++i) col_sum[i] = 0;
	for (typename SparseMatrixT1::iterator ii = a.begin(); ii != a.end(); ++ii) {
		int arow = ii.row();
		int brow = trans_row.a2b(arow);
		int acol = ii.col();
		int bcol = trans_col.a2b(acol);

		b.set(brow, bcol, ii.val(), dups);
		if (row_sum) row_sum[brow] += ii.val();
		if (col_sum) col_sum[bcol] += ii.val();
	}
}
// ----------------------------------------------------

}	// namespace giss
