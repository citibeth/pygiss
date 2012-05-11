#include "Snowdrift.hpp"
#include "eqp_x.hpp"
#include "qpt_x.hpp"

#include "Python.h"
extern PyTypeObject SnowdriftType;

namespace giss {

/** Reads from a netCDF file */
Snowdrift::Snowdrift(std::string const &fname)
 : proj_area1hp(overlap_area1hp)
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
	grid1(std::move(_grid1)), grid2(std::move(_grid2)), overlap(std::move(_overlap)), proj_area1hp(overlap_area1hp)
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
printf("grid1->size() == %ld\n", grid1->size());
printf("grid2->size() == %ld\n", grid2->size());
	n1h = n1 * num_hclass;

	// Create a height-classified overlap matrix
	overlaph.reset(new VectorSparseMatrix(SparseDescr(
		n1h, n2, overlap->index_base)));

	// Construct height-classified overlap matrix
	std::set<int> used1h_set;	// Grid cells used in overlap matrix
	std::set<int> used2_set;

printf("overlap->size() = %ld, index_base=%d\n", overlap->size(), overlap->index_base);
	for (auto oi = overlap->begin(); oi != overlap->end(); ++oi) {
		int i2 = oi.col();
		if (!mask2(i2)) continue;	// Exclude this cell

		int i1 = oi.row();

//if (i1>168857) printf("i1=%d, i2=%d\n", i1, i2);

		// Determine which height class this small grid cell is in,
		// (or at least, the portion that overlap the big grid cell)
		// Element should be in range [bot, top)

		// upper_bound returns first item > ele
		int hc = height_classifier.get_hclass(i1, elevation2(i2));

		// Determine the height-classified grid cell in grid1h
		int i1h = i1 * num_hclass + hc;
//printf("i1=%d, hc=%d, i1h=%d, num_hclass=%d\n", i1, hc, i1h, num_hclass);

		// Store it in the height-classified overlap matrix
		overlaph->set(i1h, i2, oi.val());
		used1h_set.insert(i1h);
//printf("used2_set.insert(%07d)\n", i2);
		used2_set.insert(i2);
	}

	n1hp = used1h_set.size();
	n2p = used2_set.size();

	// Set up vectors for scatter-gather of subselected matrix
	// Gather value for subselected matrix
	_i1h_to_i1hp.clear(); _i1h_to_i1hp.resize(n1h, -1);
	_i1hp_to_i1h.clear(); _i1hp_to_i1h.reserve(n1hp);
printf("n1h=%d\n", n1h);
printf("n1hp=%d\n", n1hp);
	for (auto i1h = used1h_set.begin(); i1h != used1h_set.end(); ++i1h) {
		int i1hp = _i1hp_to_i1h.size();
//printf("(i1h, i1hp) = (%d, %d)\n", *i1h, i1hp);
		_i1hp_to_i1h.push_back(*i1h);
		_i1h_to_i1hp[*i1h] = i1hp;
	}


	_i2_to_i2p.clear(); _i2_to_i2p.resize(n2, -1);
	_i2p_to_i2.clear(); _i2p_to_i2.reserve(n2p);
	//Z2p.reserve(n2p);
	for (auto i2 = used2_set.begin(); i2 != used2_set.end(); ++i2) {
		int i2p = _i2p_to_i2.size();
		_i2p_to_i2.push_back(*i2);
		_i2_to_i2p[*i2] = i2p;
	}

	// Get smoothing matrix (sub-select it later)
printf("used2_set.size() == %ld\n", used2_set.size());
	std::unique_ptr<Grid::SmoothingFunction> smooth2(grid2->get_smoothing_function(used2_set));

	// Allocate for our QP problem
printf("overlaph=%p, smooth2=%p\n", &*overlaph, &*smooth2);
printf("overlaph->size() = %ld\n", overlaph->size());
printf("smooth2->size() = %ld\n", smooth2->H.size());
	prob.reset(new QPT_problem(n1hp, n2p, overlaph->size(), smooth2->H.size(), 1));
	overlaphp.reset(new ZD11SparseMatrix(prob->A, 0));
	smooth2p_H.reset(new ZD11SparseMatrix(prob->H, 0));
	smooth2p_G0.clear(); smooth2p_G0.resize(n2p);

	// Set up the simple things
	infinity = 1e20;
//	prob->f = 0;				// Constant term of objective function
	for (int i2p=0; i2p<n2p; ++i2p) {
//		prob->X_l[i2p] = -infinity;		// Lower bound for result
		prob->X_l[i2p] = 0;		// Lower bound for result
		prob->X_u[i2p] = infinity;		// Upper bound for result
	}
//	for (int i2p=0; i2p < n2p; ++i2p) {
//		prob->G[i2p] = 0;		// Linear term of objective function
//	}


	// Subselect the smoothing function
	for (auto ii = smooth2->H.begin(); ii != smooth2->H.end(); ++ii) {
		int row2p = i2_to_i2p(ii.row());
		int col2p = i2_to_i2p(ii.col());
		smooth2p_H->set(row2p, col2p, ii.val());
	}
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		smooth2p_G0[i2p] = smooth2->G0[i2];
	}
	smooth2.reset();

	// Subselect the overlap matrix.  Also, compute row and column sums
	overlap_area1hp.clear(); overlap_area1hp.resize(n1hp, 0);
	overlap_area2p.clear(); overlap_area2p.resize(n2p, 0);
	for (auto oi = overlaph->begin(); oi != overlaph->end(); ++oi) {
		int i1hp = i1h_to_i1hp(oi.row());
		int i2p = i2_to_i2p(oi.col());
		overlaphp->set(i1hp, i2p, oi.val());

		overlap_area1hp[i1hp] += oi.val();
		overlap_area2p[i2p] += oi.val();
	}

	// Compute native area by comparing to original cells
	native_area1hp.clear(); native_area1hp.reserve(n1hp);
	for (int i1hp=0; i1hp<n1hp; ++i1hp) {
		int i1h = i1hp_to_i1h(i1hp);
		int i1 = i1h_to_i1(i1h);
		GridCell const &gc1((*grid1).operator[](i1));

		native_area1hp[i1hp] = overlap_area1hp[i1hp] * (gc1.native_area / gc1.proj_area);
	}

	// Set up GridCell pointers for grid2p
	grid2p.clear();
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GridCell const &gc2((*grid2).operator[](i2));
		grid2p.push_back(&gc2);
	}

#if 0	// Not needed
	native_area2p.clear(); native_area2p.reserve(n2p);
	for (int i2p=0; i2p < n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GridCell const &gc2(grid2[i2 + grid2->index_base]);

		native_area2p[i2p] = overlap_area2p[i2p] * (gc2.native_area / gc2.proj_area);
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
	std::vector<double> Z1hp;
	Z1hp.reserve(n1hp);
	for (int i1hp=0; i1hp<n1hp; ++i1hp) {
		int i1h = i1hp_to_i1h(i1hp);
		int i1 = i1h_to_i1(i1h);
		int hclass = get_hclass(i1h, i1);

		Z1hp.push_back(Z1[hclass](i1) *
			(native_area1hp[i1hp] / proj_area1hp[i1hp]));
	}

	// Compute |Z1hp|
	double Z1hp_sum = 0;
	for (int i1hp=0; i1hp < n1hp; ++i1hp) Z1hp_sum += overlap_area1hp[i1hp] * Z1hp[i1hp];
	printf("downgrid: |Z1hp| = %g\n", Z1hp_sum);

	// Get initial approximation of answer (that satisfies our constraints)
	double * const Z2p(prob->X);
	overlaphp->multiplyT(&Z1hp[0], Z2p);
	for (int i2p=0; i2p < n2p; ++i2p) {
		GridCell const &gc2(*grid2p[i2p]);
		Z2p[i2p] *= gc2.proj_area / (overlap_area2p[i2p] * gc2.native_area);
	}

	// Compute |Z2p|
	double Z2p_sum = 0;
	for (int i2p=0; i2p < n2p; ++i2p) Z2p_sum += overlap_area2p[i2p] * Z2p[i2p];
	printf("downgrid: |Z2p| initial = %g\n", Z2p_sum);

	// Re-arrange mass within each GCM gridcell (grid1)
	if (use_snowdrift) {
		// Zero things out
		for (int i1hp=0; i1hp<n1hp; ++i1hp) {
			prob->Y[i1hp] = 0;
		}
		for (int i2p=0; i2p<n2p; ++i2p) {
			prob->Z[i2p] = 0;
		}

		// Linear and constant terms of objective function
		// This is based on get_smoothing_function().
		// If x1 is missing, then instead of (x0-x1)**2 to the objective
		// function, we want to add (x0-x0bar)**2 where x0bar is the
		// result of the HNTR regridding for gridcell 0
		prob->f = 0;
		for (int i2p=0; i2p<n2p; ++i2p) {
			double weight = smooth2p_G0[i2p];
			prob->G[i2p] = -2.0 * weight * Z2p[i2p];
			prob->f     +=        weight * Z2p[i2p]*Z2p[i2p];
		}

printf("prob = %p\n", &*prob);
printf("prob->C = %p\n", prob->C);
		// RHS of equality constraints
		for (int i1hp=0; i1hp<n1hp; ++i1hp) {
			double val = Z1hp[i1hp] * overlap_area1hp[i1hp];

			// We have "-val" here because the constraint is phrased as "Ax + c = 0"
			// See the GALAHAD documentation eqp.pdf
			prob->C[i1hp] = -val;
			// prob->C_l[i1hp] = -val;
			// prob->C_u[i1hp] = -val;
		}

		//for (int i2p=0; i2p<n2p; ++i2p) {
		//	prob->X[i2p] = Z2p[i2p];
		//}

		// Solve the QP problem
		ret = eqp_solve_simple_(prob->main, infinity);

		// Compute |Z2p|
		double Z2p_sum = 0;
		for (int i2p=0; i2p < n2p; ++i2p) Z2p_sum += overlap_area2p[i2p] * Z2p[i2p];
		printf("downgrid: |Z2p| final = %g\n", Z2p_sum);
	}

//	store_result(_i2p_to_i2, Z2, Z2p,
//		&overlap_area2p[0], &grid2p[0], merge_or_replace);

#if 1
	// Merge results into Z2
	for (int i2p=0; i2p < n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		// double X0 = Z2(i2);		// Original value

		GridCell const &gc2(*grid2p[i2p]);
		double X1 = Z2p[i2p] * (gc2.proj_area / gc2.native_area);	// Regrid result

		switch(merge_or_replace) {
			case MergeOrReplace::MERGE : {
				double overlap_fraction = overlap_area2p[i2p] / gc2.proj_area;
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

	// Select out Z, and scale to projected space
	double Z2p_sum_proj = 0;
	double Z2p_sum_native = 0;
	std::vector<double> Z2p;
	Z2p.reserve(n2p);
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GridCell const &gc2(*grid2p[i2p]);
		double factor = (gc2.native_area / gc2.proj_area);
		Z2p.push_back(Z2(i2) * factor);
		Z2p_sum_proj += overlap_area2p[i2p] * Z2(i2);
		Z2p_sum_native += overlap_area2p[i2p] * Z2(i2) * factor;
	}

	// Print |Z2p|
	printf("upgrid: |Z2p| = %g\n", Z2p_sum_native);

	// Regrid by multiplying by overlap matrix (HNTR regridding)
	double Z1hp[n1hp];
	overlaphp->multiply(&Z2p[0], Z1hp);

	// Scale back to Z/m^2
	for (int i1hp=0; i1hp < n1hp; ++i1hp) {
		Z1hp[i1hp] /= proj_area1hp[i1hp];		// Because proj_area1hp == overlap_area1hp
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
if (i1h < 5) printf("i1h=%d, i1=%d\n", i1h, i1);
		int hclass = get_hclass(i1h, i1);
		// double X0 = Z1[hclass](i1);

		double X1 = Z1hp[i1hp] * (proj_area1hp[i1hp] / native_area1hp[i1hp]);	// Regrid result

		// Merge makes no sense here, because by definition,
		// proj_area1hp == overlap_area1hp.  So merge just reduces
		// to replace
		Z1[hclass](i1) = X1;
		Z1hp_sum_native += X1 * native_area1hp[i1hp];
	}
	printf("upgrid: |Z1hp| = %g\n", Z1hp_sum_native);


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
