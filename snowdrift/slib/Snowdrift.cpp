#include "Snowdrift.hpp"
#include "eqp_x.hpp"
#include "qpt_x.hpp"

#include "Python.h"
extern PyTypeObject SnowdriftType;

namespace giss {

/** Reads from a netCDF file */
Snowdrift::Snowdrift(std::string const &fname)
{
	NcFile nc(fname.c_str());
	grid1 = Grid::netcdf_read(nc, "grid1");
	grid2 = Grid::netcdf_read(nc, "grid2");
	overlap = VectorSparseMatrix::netcdf_read(nc, "overlap");
	nc.close();

	n1 = grid1->max_index - grid1->index_base + 1;
	n2 = grid2->max_index - grid2->index_base + 1;
}



Snowdrift::Snowdrift(
std::unique_ptr<Grid> &&_grid1,
std::unique_ptr<Grid> &&_grid2,
std::unique_ptr<VectorSparseMatrix> &&_overlap) :
	grid1(std::move(_grid1)), grid2(std::move(_grid2)), overlap(std::move(_overlap))
{
	n1 = grid1->max_index - grid1->index_base + 1;
	n2 = grid2->max_index - grid2->index_base + 1;
}




/**
@param overlap [n1 x n2] Overlap matrix between grid1 and grid2
@param mask2 [n2] (bool) Which grid cells in grid2 we want to actually use
@param elevation2 [n2] Topography
@param height_max1 [num_hclass][n1] Height class boundaries for each cell in grid1.
                   From LISMB_COM.F90 in ModelE.
*/
void Snowdrift::init(
blitz::Array<double,1> const &elevation2,
blitz::Array<int,1> const &mask2,
HeightClassifier &height_classifier)
{
	num_hclass = height_classifier.num_hclass();
	n1h = n1 * num_hclass;

	// ========= Construct overlaph: height-classified overlap matrix
	std::set<int> used1h_set;	// Grid cells used in overlap matrix
	std::set<int> used2_set;

	std::unique_ptr<VectorSparseMatrix> overlaph(new VectorSparseMatrix(SparseDescr(
		n1h, n2, overlap->index_base)));

	for (auto oi = overlap->begin(); oi != overlap->end(); ++oi) {
		int i2 = oi.col();
		if (!mask2(i2)) continue;	// Exclude this cell

		int i1 = oi.row();

		// Determine which height class this small grid cell is in,
		// (or at least, the portion that overlap the big grid cell)
		// Element should be in range [bot, top)

		// upper_bound returns first item > ele
		int hc = height_classifier.get_hclass(i1, elevation2(i2));

		// Determine the height-classified grid cell in grid1h
		int i1h = i1 * num_hclass + hc;

		// Store it in the height-classified overlap matrix
		overlaph->set(i1h, i2, oi.val());
		used1h_set.insert(i1h);
		used2_set.insert(i2);
	}

	// ========== Construct translator between (full) and (*p) spaces
	n1hp = used1h_set.size();
	n2p = used2_set.size();

	// Set up vectors for scatter-gather of subselected matrix
	// Gather value for subselected matrix
	_i1h_to_i1hp.clear(); _i1h_to_i1hp.resize(n1h, -1);
	_i1hp_to_i1h.clear(); _i1hp_to_i1h.reserve(n1hp);
	for (auto i1h = used1h_set.begin(); i1h != used1h_set.end(); ++i1h) {
		int i1hp = _i1hp_to_i1h.size();
		_i1hp_to_i1h.push_back(*i1h);
		_i1h_to_i1hp[*i1h] = i1hp;
	}

	_i2_to_i2p.clear(); _i2_to_i2p.resize(n2, -1);
	_i2p_to_i2.clear(); _i2p_to_i2.reserve(n2p);
	for (auto i2 = used2_set.begin(); i2 != used2_set.end(); ++i2) {
		int i2p = _i2p_to_i2.size();
		_i2p_to_i2.push_back(*i2);
		_i2_to_i2p[*i2] = i2p;
	}

	// ========== Construct overlaphp
	// Compute overlaphp (subselect overlaph)
	overlaphp.reset(new VectorSparseMatrix(SparseDescr(
		n1hp, n2p, overlap->index_base)));
	overlap_area1hp.clear(); overlap_area1hp.resize(n1hp, 0);
	overlap_area2p.clear(); overlap_area2p.resize(n2p, 0);
	std::vector<int> row_count1hp(n1hp);	// For computation of *q space below.
	for (auto oi = overlaph->begin(); oi != overlaph->end(); ++oi) {
		int i1h = oi.row();

		int i1hp = i1h_to_i1hp(i1h);
		int i2p = i2_to_i2p(oi.col());
		overlaphp->set(i1hp, i2p, oi.val());

		overlap_area1hp[i1hp] += oi.val();
		++row_count1hp[i1hp];
		overlap_area2p[i2p] += oi.val();
	}

	// ========== Set up gridcell pointers for *p space
	grid2p.clear();  grid2p.reserve(n2p);
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GridCell const &gc2((*grid2).operator[](i2));
		grid2p.push_back(GCInfo(gc2.native_area, gc2.proj_area));
	}

	// Compute native area by comparing to original cells
	grid1hp.clear(); grid1hp.reserve(n1hp);
	for (int i1hp=0; i1hp<n1hp; ++i1hp) {
		int i1h = i1hp_to_i1h(i1hp);
		int i1 = i1h_to_i1(i1h);
		GridCell const &gc1((*grid1).operator[](i1));

		double proj_area = overlap_area1hp[i1hp];
		double native_area = overlap_area1hp[i1hp] * (gc1.native_area / gc1.proj_area);
		grid1hp.push_back(GCInfo(native_area, proj_area));
	}

	// ================= Set up *q space (subspace of *p space)

	// Eliminate GCM grid cells with just one ice grid cell intersection.
	// Removes columns that would generate too few terms in the constraints matrix
	// This is necessary to keep certain matrices positive-definite in the EQP solver

	// Determine indices used in the *q space
	std::set<int> delete1h_set;
	std::set<int> delete2_set;
	int const min_row_count = 2;
//	int const min_row_count = 100000;
	int overlaphq_size = 0;
	for (auto oi = overlaphp->begin(); oi != overlaphp->end(); ++oi) {
		int i1hp = oi.row();
		int i1h = i1hp_to_i1h(i1hp);
		int i2p = oi.col();
		int i2 = i2p_to_i2(i2p);
		if (row_count1hp[i1hp] < min_row_count) {
			delete1h_set.insert(i1h);
			delete2_set.insert(i2);
		}
	}

	// Count number of elements in overlaphq, and get definitive set of rows and cols
	used1h_set.clear();			// Indies in (full) space
	used2_set.clear();
	for (auto oi = overlaphp->begin(); oi != overlaphp->end(); ++oi) {
		int i1hp = oi.row();
		int i1h = i1hp_to_i1h(i1hp);
		int i2p = oi.col();
		int i2 = i2p_to_i2(i2p);

		if (delete1h_set.find(i1h) != delete1h_set.end()) continue;
		if (delete2_set.find(i2) != delete2_set.end()) continue;

		++overlaphq_size;		// Count so we know how much to allocate below
		used1h_set.insert(i1h);
		used2_set.insert(i2);
	}
	n1hq = used1h_set.size();
	n2q = used2_set.size();

	_i1hp_to_i1hq.clear(); _i1hp_to_i1hq.resize(n1hp, -1);
	_i1hq_to_i1hp.clear(); _i1hq_to_i1hp.reserve(n1hq);
	for (auto i1h = used1h_set.begin(); i1h != used1h_set.end(); ++i1h) {
		int i1hp = i1h_to_i1hp(*i1h);
		int i1hq = _i1hq_to_i1hp.size();
		_i1hq_to_i1hp.push_back(i1hp);
		_i1hp_to_i1hq[i1hp] = i1hq;
	}

	_i2p_to_i2q.clear(); _i2p_to_i2q.resize(n2p, -1);
	_i2q_to_i2p.clear(); _i2q_to_i2p.reserve(n2q);
	for (auto i2 = used2_set.begin(); i2 != used2_set.end(); ++i2) {
		int i2p = i2_to_i2p(*i2);
		int i2q = _i2q_to_i2p.size();
		_i2q_to_i2p.push_back(i2p);
		_i2p_to_i2q[i2p] = i2q;
	}

	// ------- Allocate our problem for GALAHAD (to be filled in later...)
	// Get smoothing matrix (sub-select it later)
	std::unique_ptr<Grid::SmoothingFunction> smooth2(grid2->get_smoothing_function(used2_set));
	prob.reset(new QPT_problem(n1hq, n2q, overlaphq_size, smooth2->H.size(), 1));
	overlaphq.reset(new ZD11SparseMatrix(prob->A, 0));

	// -------- Separate overlaphp into overlaphq and overlaphp_extra
//	overlaphp_extra.reset(new VectorSparseMatrix(SparseDescr(
//		n1hp, n2p, overlaphp->index_base)));
	overlap_area1hq.clear(); overlap_area1hq.resize(n1hq, 0);
	overlap_area2q.clear(); overlap_area2q.resize(n2q, 0);
	for (auto oi = overlaphp->begin(); oi != overlaphp->end(); ++oi) {
		int i1hp = oi.row();
		int i1h = i1hp_to_i1h(i1hp);
		int i2p = oi.col();
		int i2 = i2p_to_i2(i2p);

		if (delete1h_set.find(i1h) != delete1h_set.end()) continue;
		if (delete2_set.find(i2) != delete2_set.end()) continue;

//		if (row_count1hp[i1hp] >= min_row_count) {
			int i1hq = i1hp_to_i1hq(i1hp);	// Shouldn't be <0
			int i2q = i2p_to_i2q(i2p);
			overlaphq->set(i1hq, i2q, oi.val());
			overlap_area1hq[i1hq] += oi.val();
			overlap_area2q[i2q] += oi.val();
//		} else {
//			printf("Excluding (%d,%d) area=%f from overlaphq\n", i1hp, i2p, oi.val());
//		}
	}

	// ------- Renumber rows and cols on smoothing function, for *q space
	smooth2q_H.reset(new ZD11SparseMatrix(prob->H, 0));
	smooth2q_G0.clear(); smooth2q_G0.resize(n2q);
	for (auto ii = smooth2->H.begin(); ii != smooth2->H.end(); ++ii) {
		int row2q = i2p_to_i2q(i2_to_i2p(ii.row()));
		int col2q = i2p_to_i2q(i2_to_i2p(ii.col()));
		smooth2q_H->set(row2q, col2q, ii.val());
	}
	for (int i2q=0; i2q<n2q; ++i2q) {
		int i2p = i2q_to_i2p(i2q);
		int i2 = i2p_to_i2(i2p);
		smooth2q_G0[i2q] = smooth2->G0[i2];
	}
	smooth2.reset();


	// ============== Set up other things for the QP problem
	// Set up the simple things
	infinity = 1e20;
#if 0	// This only counts for QP, not EQP
	for (int i2q=0; i2q<n2q; ++i2q) {
		prob->X_l[i2q] = -infinity;		// Lower bound for result
		prob->X_u[i2q] = infinity;		// Upper bound for result
	}
#endif

}


/**
@param Z1 Field to downgrid. [num_hclass][n1]
@param Z2 output, for regridded data.
@param merge true to merge in results, false for replace
*/
bool Snowdrift::downgrid(
std::vector<blitz::Array<double,1>> const &Z1,
blitz::Array<double,1> &Z2,
MergeOrReplace merge_or_replace,
bool use_snowdrift)
{
	bool ret = true;

	// Select out Z
	std::vector<double> Z1hp; Z1hp.reserve(n1hp);
	for (int i1hp=0; i1hp<n1hp; ++i1hp) {
		int i1h = i1hp_to_i1h(i1hp);
		int i1 = i1h_to_i1(i1h);
		int hclass = get_hclass(i1h, i1);

		GCInfo const &gc1h(grid1hp[i1hp]);

		Z1hp.push_back(Z1[hclass](i1) * gc1h.native_by_proj());
	}

	// Compute |Z1hp|
	double Z1hp_sum = 0;
	for (int i1hp=0; i1hp < n1hp; ++i1hp) Z1hp_sum += overlap_area1hp[i1hp] * Z1hp[i1hp];
	printf("downgrid: |Z1hp| = %1.15g\n", Z1hp_sum);

	// Get initial approximation of answer (that satisfies our constraints)
	std::vector<double> Z2p(n2p);
	overlaphp->multiplyT(&Z1hp[0], &Z2p[0]);
	for (int i2p=0; i2p < n2p; ++i2p) {
		GCInfo const &gc2(grid2p[i2p]);
//		Z2p[i2p] *= gc2.proj_area / (overlap_area2p[i2p] * gc2.native_area);
		Z2p[i2p] /= overlap_area2p[i2p];
	}

	// Compute |Z2p|
	double Z2p_sum = 0;
	for (int i2p=0; i2p < n2p; ++i2p) {
		GCInfo const &gc2(grid2p[i2p]);
		Z2p_sum += overlap_area2p[i2p] * Z2p[i2p] * gc2.proj_by_native();
	}
	printf("downgrid: |Z2p| initial = %1.15g\n", Z2p_sum);

	// Re-arrange mass within each GCM gridcell (grid1)
	if (use_snowdrift) {
		// Subselect out Z2q
		double * const Z2q(prob->X);
		for (int i2q=0; i2q < n2q; ++i2q) {
			int i2p = i2q_to_i2p(i2q);
			Z2q[i2q] = Z2p[i2p];
		}

		// Zero things out for QP algo
		for (int i1hq=0; i1hq<n1hq; ++i1hq) {
			prob->Y[i1hq] = 0;
		}
		for (int i2q=0; i2q<n2q; ++i2q) {
			prob->Z[i2q] = 0;
		}

		// Linear and constant terms of objective function
		// This is based on get_smoothing_function().
		// If x1 is missing, then instead of (x0-x1)**2 to the objective
		// function, we want to add (x0-x0bar)**2 where x0bar is the
		// result of the HNTR regridding for gridcell 0
		prob->f = 0;
		for (int i2q=0; i2q<n2q; ++i2q) {
			double weight = smooth2q_G0[i2q];
			prob->G[i2q] = -2.0 * weight * Z2q[i2q];
			prob->f     +=        weight * Z2q[i2q]*Z2q[i2q];
		}

printf("prob = %p\n", &*prob);
printf("prob->C = %p\n", prob->C);
		// RHS of equality constraints
printf("*** n1hp=%d, n1hq=%d\n", n1hp, n1hq);
		for (int i1hq=0; i1hq<n1hq; ++i1hq) {
			int i1hp = i1hq_to_i1hp(i1hq);
			double val = Z1hp[i1hp] * overlap_area1hq[i1hq];
//			double val = Z1hp[i1hp] * overlap_area1hp[i1hp];
double frac = (overlap_area1hq[i1hq] - overlap_area1hp[i1hp]) / overlap_area1hp[i1hp];
//if (std::abs(frac) > 1e-14) {
if (overlap_area1hq[i1hq] != overlap_area1hp[i1hp]) {
	printf("i1hq=%d, i1hp=%d, o1hq=%f, o1hp=%f\n", i1hq, i1hp, overlap_area1hq[i1hq], overlap_area1hp[i1hp]);
}

			// We have "-val" here because the constraint is phrased as "Ax + c = 0"
			// See the GALAHAD documentation eqp.pdf
			prob->C[i1hq] = -val;
			// prob->C_l[i1hq] = -val;
			// prob->C_u[i1hq] = -val;
		}
printf("*** done\n");

		// Solve the QP problem
		ret = eqp_solve_simple_(prob->main, infinity);

		// Put answer back int Z2p (leaving cells we're not responsible for alone)
		for (int i2q=0; i2q < n2q; ++i2q) {
			int i2p = i2q_to_i2p(i2q);
			Z2p[i2p] = Z2q[i2q];
		}
#if 0
			double overlap_q = overlap_area2q[i2q];
			double overlap_p = overlap_area2p[i2p];
if (overlap_p != overlap_q) printf("i2q=%d, overlap_q=%f, overlap_p=%f\n", i2q, overlap_q, overlap_p);

			double stuff = Z2p[i2p] * (overlap_p - overlap_q) + Z2q[i2q] * overlap_q;
			Z2p[i2p] = stuff / overlap_p;

//			// overlap_p contains overlap_q
//			double q_fraction = overlap_q / overlap_p;
//			double p_fraction = 1.0 - q_fraction;
//			Z2p[i2p] = (p_fraction * Z2p[i2p]) + (q_fraction * Z2q[i2q]);
		}
#endif

		// Compute |Z2p|
		double Z2p_sum = 0;
		for (int i2p=0; i2p < n2p; ++i2p) Z2p_sum += overlap_area2p[i2p] * Z2p[i2p];
		printf("downgrid: |Z2p| final = %1.15g\n", Z2p_sum);
	}

//	store_result(_i2p_to_i2, Z2, Z2p,
//		&overlap_area2p[0], &grid2p[0], merge_or_replace);

#if 1
	// Merge results into Z2
	for (int i2p=0; i2p < n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);

		GCInfo const &gc2(grid2p[i2p]);
		double X1 = Z2p[i2p] * (gc2.proj_area / gc2.native_area);	// Regrid result

		switch(merge_or_replace) {
			case MergeOrReplace::MERGE : {
				double overlap_fraction = overlap_area2p[i2] / gc2.proj_area;
				Z2(i2) =
					(1.0 - overlap_fraction) * Z2(i2) +		// Original value
					overlap_fraction * X1;					// Our new value
			} break;
			default :		// REPLACE
				Z2(i2) = X1;
			break;
		}
	}
#endif

	fflush(stdout);
	return ret;
}

/**
@param Z2 Field to upgrid. [n2]
@param Z1 Regridded outpu. [num_hclass][n1]
@param merge true to merge in results, false for replace
*/
void Snowdrift::upgrid(
blitz::Array<double,1> const &Z2,
std::vector<blitz::Array<double,1>> &Z1,
MergeOrReplace merge_or_replace)
{

printf("Snowdrift::upgrid(merge_or_replace = %d)\n", merge_or_replace);

	// This could all be done in-place by inlining the multiplication function...

	// Select out Z, and scale to projected space
	double Z2p_sum_proj = 0;
	double Z2p_sum_native = 0;
	std::vector<double> Z2p;
	Z2p.reserve(n2p);
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GCInfo const &gc2(grid2p[i2p]);
		double factor = (gc2.native_area / gc2.proj_area);
		Z2p.push_back(Z2(i2) * factor);
		Z2p_sum_proj += overlap_area2p[i2p] * Z2(i2);
		Z2p_sum_native += overlap_area2p[i2p] * Z2(i2) * factor;
	}

	// Print |Z2p|
	printf("upgrid: |Z2p| = %1.15g\n", Z2p_sum_native);

	// Regrid by multiplying by overlap matrix (HNTR regridding)
	double Z1hp[n1hp];
	overlaphp->multiply(&Z2p[0], Z1hp);

	// Scale back to Z/m^2
	for (int i1hp=0; i1hp < n1hp; ++i1hp) {
		Z1hp[i1hp] /= overlap_area1hp[i1hp];
	}


	// Compute |Z1hp|
//	double Z1hp_sum = 0;
//	for (int i1hp=0; i1hp < n1hp; ++i1hp) Z1hp_sum += overlap_area1hp[i1hp] * Z1hp[i1hp];
//	printf("upgrid: |Z1hp| = %g\n", Z1hp_sum);

	// Merge results into Z1
	double Z1hp_sum_native = 0;
	for (int i1hp=0; i1hp < n1hp; ++i1hp) {
		int i1h = i1hp_to_i1h(i1hp);
		int i1 = i1h_to_i1(i1h);
		int hclass = get_hclass(i1h, i1);
		// double X0 = Z1[hclass](i1);
		GCInfo const &gc1h(grid1hp[i1hp]);

		double X1 = Z1hp[i1hp] * (gc1h.proj_area / gc1h.native_area);	// Regrid result

		// Merge makes no sense here, because by definition,
		// proj_area1hp == overlap_area1hp.  So merge just reduces
		// to replace
		Z1[hclass](i1) = X1;
		Z1hp_sum_native += X1 * gc1h.native_area;
	}
	printf("upgrid: |Z1hp| = %1.15g\n", Z1hp_sum_native);


}








// /** Stores result of a regridding back into the main field.  Does two things:
//  a) Scatter of result
//  b) Merge cells of partial overlap into existing values, if needed.
// 
// @param i2p_to_i2 Translate indices from regridded result to main field
// @param Z2 Original value
// @param Z2p Result of regridding
// */
// void store_result(
// std::vector<int> const &i2p_to_i2,
// blitz::Array<double,1> &Z2, double const *Z2p,
// double const *overlap_area2p,
// GridCell const **grid2p,
// MergeOrReplace merge_or_replace)
// {
// 	int n2p = i2p_to_i2.size();
// 
// 	// Merge results into Z2
// 	for (int i2p=0; i2p < n2p; ++i2p) {
// 		int i2 = i2p_to_i2[i2p];
// 		double X0 = Z2[i2];		// Original value
// 		double X1 = Z2p[i2p];	// Result of simple regridding
// 
// 		GridCell const &gc2(*grid2p[i2p]);
// 		double X1 = Z2p[i2p] * (gc2.proj_area / gc2.native_area);	// Regrid result
// 
// 		switch(merge_or_replace) {
// 			case MergeOrReplace::MERGE :
// 				double overlap_fraction = overlap_area2p[i2p] / gc2.proj_area;
// 				Z2[i2] =
// 					(1.0 - overlap_fraction) * Z2[i2] +		// Original value
// 					overlap_fraction * X1;					// Our new value
// 			break;
// 			default :		// REPLACE
// 				Z2[i2] = X1;
// 		}
// 	}
// }
// 

}	// namespace giss
