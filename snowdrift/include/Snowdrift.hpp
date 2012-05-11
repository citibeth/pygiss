#pragma once

#include <exception>
#include <memory>
#include <blitz/array.h>
#include "qpt_x.hpp"

#include "Grid.hpp"
#include "SparseMatrix.hpp"
#include "HeightClassifier.hpp"

namespace giss {

/**
<pre>The following suffixes are used in variable names
    Suffix      Description
    ======      ==================================
    1           Original grid1 (GCM grid)
    2           Original grid2 (ice grid)
    1h          Height-classified grid1
    1hp         Sub-selected grid1h (zero rows & cols removed)
    2p          Sub-selected grid2
</pre>
*/
class Snowdrift {
public:
	int num_hclass;

	enum class MergeOrReplace {MERGE, REPLACE};

	// ------- Original Grids and overlap matrix read from netCDF file
	std::unique_ptr<Grid> grid1;
	std::unique_ptr<Grid> grid2;
	std::unique_ptr<VectorSparseMatrix> overlap;	/// Overlap matrix between grid1 and grid2

	// -------- Arguments to regrid function
	std::vector<blitz::Array<double,1>> const Z1;	// Z1[heighti][i1]

	// ------ Additional stuff we compute in preparation for regridding
	std::unique_ptr<QPT_problem> prob;	// Data structure for QP solver
	double infinity;					// Value used for infinity in QP solver
	std::unique_ptr<ZD11SparseMatrix> overlaphp;	// Overlap (constratints) matrix used for QP problem
	std::unique_ptr<ZD11SparseMatrix> smooth2p_H;		// Hessian for QP solver
	std::vector<double> smooth2p_G0;					// Subselected version of smooth2.G0

	int n1;		// grid1.size() = overlap.nrow
	int n2;		// grid2.size() = overlap.ncol
	int n1h;	// grid1h.size() = overlaph.nrow
	int n1hp;	// grid1hp.size() = overlaphp.nrow
	int n2p;	// grid2.size() = overlaphp.ncol

	/// Overlap matrix between grid1h and grid2
	std::unique_ptr<VectorSparseMatrix> overlaph;
	std::vector<int> _i1h_to_i1hp;		// Converts i1h --> i1hp
	std::vector<int> _i1hp_to_i1h;			// Converts i1hp --> i1h

#if 0
	inline int i1h_to_i1hp(int i1h)
		{ return _i1h_to_i1hp[i1h]; }
	inline int i1hp_to_i1h(int i1hp)
		{ return _i1hp_to_i1h[i1hp]; }
#else
	inline int i1h_to_i1hp(int i1h)
	{
		if (i1h < 0 || i1h >= _i1h_to_i1hp.size()) {
			fprintf(stderr, "i1h=%d is out of range (%d, %d)\n", i1h, 0, _i1h_to_i1hp.size());
			throw std::exception();
		}
		int i1hp = _i1h_to_i1hp[i1h];
		if (i1hp < 0) {
			fprintf(stderr, "i1h=%d produces invalid i1hp=%d\n", i1h, i1hp);
			throw std::exception();
		}
		return i1hp;
	}
	inline int i1hp_to_i1h(int i1hp)
	{
		if (i1hp < 0 || i1hp >= _i1hp_to_i1h.size()) {
			fprintf(stderr, "i1hp=%d is out of range (%d, %d)\n", i1hp, 0, _i1hp_to_i1h.size());
			throw std::exception();
		}
		int i1h = _i1hp_to_i1h[i1hp];
		if (i1h < 0) {
			fprintf(stderr, "i1hp=%d produces invalid i1h=%d\n", i1hp, i1h);
			throw std::exception();
		}
		return i1h;
	}
#endif


	inline int i1h_to_i1(int i1h)
		{ return i1h / num_hclass; }
	inline int get_hclass(int i1h, int i1)
		{ return i1h - i1 * num_hclass; }

	std::vector<int> _i2_to_i2p;		// Converts i2 --> i2p
	std::vector<int> _i2p_to_i2;			// Converts i2p --> i2

	inline int i2_to_i2p(int i2)
	{
		if (i2 < 0 || i2 >= _i2_to_i2p.size()) {
			fprintf(stderr, "i2=%d is out of range (%d, %d)\n", i2, 0, _i2_to_i2p.size());
			throw std::exception();
		}
		int i2p = _i2_to_i2p[i2];
		if (i2p < 0) {
			fprintf(stderr, "i2=%d produces invalid i2p=%d\n", i2, i2p);
			throw std::exception();
		}
		return i2p;
	}
	inline int i2p_to_i2(int i2p)
	{
		if (i2p < 0 || i2p >= _i2p_to_i2.size()) {
			fprintf(stderr, "i2p=%d is out of range (%d, %d)\n", i2p, 0, _i2p_to_i2.size());
			throw std::exception();
		}
		int i2 = _i2p_to_i2[i2p];
		if (i2 < 0) {
			fprintf(stderr, "i2p=%d produces invalid i2=%d\n", i2p, i2);
			throw std::exception();
		}
		return i2;
	}

	std::vector<double> overlap_area1hp;	// Same as proj_area1hp.  By definition, height-classified parts of grid1 are only portions that overlap grid2
	std::vector<double> &proj_area1hp;
	std::vector<double> native_area1hp;

	std::vector<double> overlap_area2p;
	// std::vector<double> native_area2p;

	std::vector<GridCell const *> grid2p;

public:
	// ---------------- Methods...

	/** Loads basic grid and overlap info from a NetCDF file */
	Snowdrift(std::string const &fname);

	Snowdrift(
	std::unique_ptr<Grid> &&_grid1,
	std::unique_ptr<Grid> &&_grid2,
	std::unique_ptr<VectorSparseMatrix> &&_overlap);

	/**
	@param overlap [n1 x n2] Overlap matrix between grid1 and grid2
	@param mask2 [n2] (bool) Which grid cells in grid2 we want to actually use
	@param elevation2 [n2] Topography
	@param height_classes Height class boundaries for each cell in grid1.
	                   From LISMB_COM.F90 in ModelE.
	*/
	void init(
	blitz::Array<double,1> const &elevation2,
	blitz::Array<int,1> const &mask2,
	HeightClassifier &height_classifier);

	/**
	@param Z1 Field to downgrid. [num_hclass][n1]
	@param Z2 output, for regridded data.
	@param merge true to merge in results, false for replace
	*/
	bool downgrid(
	std::vector<blitz::Array<double,1>> const &Z1,
	blitz::Array<double,1> &Z2,
	MergeOrReplace merge_or_replace,
	bool use_snowdrift = false);

	/**
	@param Z2 Field to upgrid. [n2]
	@param Z1 Regridded outpu. [num_hclass][n1]
	@param merge true to merge in results, false for replace
	*/
	void upgrid(
	blitz::Array<double,1> const &Z2,
	std::vector<blitz::Array<double,1>> &Z1,
	MergeOrReplace merge_or_replace);

};

}	// namespace giss
