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
	1hq,2p      Additional rows/cols removed for QP.  Subset of 1hp
</pre>
*/
class Snowdrift {

	class GCInfo {
	public:
		double native_area;
		double proj_area;

		GCInfo(double _native_area, double _proj_area) :
			native_area(_native_area), proj_area(_proj_area) {}
#if 0
		GCInfo(GCInfo const &rhs) : native_area(rhs.native_area), proj_area(rhs.proj_area) {}
		GCInfo &operator=(GCInfo const &rhs) {
			native_area = rhs.native_area;
			proj_area = rhs.proj_area;
			return *this;
		}
#endif

		double native_by_proj() const { return native_area / proj_area; }
		double proj_by_native() const { return proj_area / native_area; }
	};
public:
	struct SmoothingFunction {
		/** The Hessian matrix, used in QP */
		MapSparseMatrix H;

		/** Linear term to QP objective function.  Must be pairwise-multiplied
		by Z0, the HNTR regridding result.  Constant term is also available
		from this */
		std::vector<double> G0;

		SmoothingFunction(MapSparseMatrix &&_H) : H(_H), G0(H.ncol) {}
	};

#if 0
	struct Constraints {
		VectorSparseMatrix A;	/// The constraints matrix (nconstraints x n2 --- ice grid)
		/** Right-hand-side matrix for equality constratints: */
		RHS * Z1 = constraints.
		VectorSparseMatrix RHSEQ;

	};
#endif


	int num_hclass;

	enum class MergeOrReplace {MERGE, REPLACE};

	// ------- Original Grids and overlap matrix read from netCDF file
	std::unique_ptr<Grid> grid1;
	std::unique_ptr<Grid> grid2;
	std::unique_ptr<VectorSparseMatrix> overlap;	/// Overlap matrix between grid1 and grid2

	// -------- Arguments to regrid function
	// std::vector<blitz::Array<double,1>> const Z1;	// Z1[heighti][i1]

	// ------ Additional stuff we compute in preparation for regridding
	std::unique_ptr<QPT_problem> prob;	// Data structure for QP solver
	double infinity;					// Value used for infinity in QP solver
	std::unique_ptr<VectorSparseMatrix> overlaphp;
	std::unique_ptr<ZD11SparseMatrix> constraintshq;	// Overlap (constratints) matrix used for QP problem.  Pesky cells have been removed, and rows/columns have been renumbered to remove blank rows and columns.

	/** Cells of overlaphp that were not included in constraintshq.
	This happens for rows with only 1 element in them, since they mess up the EQP solver. */
//	std::unique_ptr<VectorSparseMatrix> overlaphp_extra;

	std::unique_ptr<ZD11SparseMatrix> smooth2q_H;		// Hessian for QP solver
	std::vector<double> smooth2q_G0;					// Subselected version of smooth2.G0

	int n1;		// grid1.size() = overlap.nrow
	int n2;		// grid2.size() = overlap.ncol
	int n1h;	// grid1h.size() = overlaph.nrow
	int n1hp;	// grid1hp.size() = overlaphp.nrow
	int n2p;	// grid2.size() = overlaphp.ncol
	int n2q;
	int nconstraints;

	// If != "", write to this file just before we solve.
	std::string problem_file;

	inline int i1h_to_i1(int i1h)
		{ return i1h / num_hclass; }
	inline int get_hclass(int i1h, int i1)
		{ return i1h - i1 * num_hclass; }


	/// Overlap matrix between grid1h and grid2 (including things that were eliminated for EQP)
	// std::unique_ptr<VectorSparseMatrix> overlaph;
	IndexTranslator trans_1h_1hp;
		inline int i1h_to_i1hp(int i1h)
			{ return trans_1h_1hp.a2b(i1h); }
		inline int i1hp_to_i1h(int i1hp)
			{ return trans_1h_1hp.b2a(i1hp); }

	IndexTranslator trans_2_2p;
		inline int i2_to_i2p(int i2)
			{ return trans_2_2p.a2b(i2); }
		inline int i2p_to_i2(int i2p)
			{ return trans_2_2p.b2a(i2p); }

#if 0
	// ------------- *p <--> *q indices (subspace to Galahad-subspace)
	IndexTranslator trans_1hp_1hq;
		inline int i1hp_to_i1hq(int i1hp)
			{ return trans_1hp_1hq.a2b(i1hp); }
		inline int i1hq_to_i1hp(int i1hq)
			{ return trans_1hp_1hq.b2a(i1hq); }
#endif
	IndexTranslator trans_2_2q;

// 		inline int i2_to_i2q(int i2p)
// 			{ return trans_2_2q.a2b(i2p); }
// 		inline int i2q_to_i2p(int i2q)
// 			{ return trans_2_2q.b2a(i2q); }




	// Based on overlaps in overlaphp
	// By definition, height-classified parts of grid1 are only portions that overlap grid2
	std::vector<double> overlap_area1hp;
	std::vector<double> overlap_area2p;

	// Based on overlaps in constraintshq
	std::vector<double> overlap_area1hq;
	std::vector<double> overlap_area2q;

	std::vector<GCInfo> grid2p;
	std::vector<GCInfo> grid1hp;

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
	std::vector<blitz::Array<double,1>> &height_max,
	boost::function<std::unique_ptr<VectorSparseMatrix>
		(VectorSparseMatrix &)> const &get_constraints);

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

	// ================= Different constraints matrices
	// Controls degree of smoothing in final result.
	static std::unique_ptr<VectorSparseMatrix> get_constraints_default(VectorSparseMatrix &overlap);
	std::unique_ptr<VectorSparseMatrix> get_constraints_cesm(VectorSparseMatrix &overlap);

};

}	// namespace giss
