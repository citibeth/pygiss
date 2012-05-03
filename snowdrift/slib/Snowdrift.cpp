#include "Snowdrift.hpp"
#include "eqp_x.hpp"
#include "qpt_x.hpp"

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
}



Snowdrift::Snowdrift(
std::unique_ptr<Grid> &&_grid1,
std::unique_ptr<Grid> &&_grid2,
std::unique_ptr<VectorSparseMatrix> &&_overlap) :
	grid1(std::move(_grid1)), grid2(std::move(_grid2)), overlap(std::move(_overlap)), proj_area1hp(overlap_area1hp)
{}




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
std::vector<blitz::Array<double,1>> const &height_max1)
{
	num_hclass = height_max1.size();
	n1 = grid1->max_index - grid1->index_base + 1;
	n2 = grid2->max_index - grid2->index_base + 1;
printf("grid1->size() == %ld\n", grid1->size());
printf("grid2->size() == %ld\n", grid2->size());
	n1h = n1 * num_hclass;

	// Create a height-classified overlap matrix
	overlaph.reset(new VectorSparseMatrix(SparseDescr(
		n1h, n2, overlap->index_base,
		SparseMatrix::MatrixStructure::TRIANGULAR,
		SparseMatrix::TriangularType::LOWER)));

	// Construct height-classified overlap matrix
	overlap->sort(SparseMatrix::SortOrder::ROW_MAJOR);	// Optimize gather of height_max_ij
	std::set<int> used1h_set;	// Grid cells used in overlap matrix
	std::set<int> used2_set;
	double height_max_ij[num_hclass+1];
	int i1_last = -1;

	for (auto oi = overlap->begin(); oi != overlap->end(); ++oi) {
		int i2 = oi.col();
		if (mask2(i2) != 0) continue;	// Exclude this cell
		double ele = elevation2(i2);

		int i1 = oi.row();

		// Gather height classes from disparate Fortran data structure
		if (i1 != i1_last) {
			for (int heighti=0; heighti<num_hclass; ++heighti)
				height_max_ij[heighti] = height_max1[heighti](i1);
			height_max_ij[num_hclass] = std::numeric_limits<double>::infinity();
			i1_last = i1;
		}

		// Determine which height class this small grid cell is in,
		// (or at least, the portion that overlap the big grid cell)
		// Element should be in range [bot, top)

		// upper_bound returns first item > ele
		double *top = std::upper_bound(height_max_ij, height_max_ij + num_hclass+1, ele);
		int hc = top - height_max_ij;
		if (hc < 0) hc = 0;		// Redundant
		if (hc >= num_hclass) hc = num_hclass - 1;

		// Determine the height-classified grid cell in grid1h
		int i1h = i1 * num_hclass + hc;

		// Store it in the height-classified overlap matrix
		overlaph->set(i1h, i2, oi.val());
		used1h_set.insert(i1h);
		used2_set.insert(i2);
	}

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
	//Z2p.reserve(n2p);
	for (auto i2 = used2_set.begin(); i2 != used2_set.end(); ++i2) {
		int i2p = _i2p_to_i2.size();
		_i2p_to_i2.push_back(*i2);
		_i2_to_i2p[*i2] = i2p;
	}

	// Get smoothing matrix (sub-select it later)
	std::unique_ptr<MapSparseMatrix> smooth2(grid2->get_smoothing_matrix(used2_set));

	// Allocate for our QP problem
	prob.reset(new QPT_problem(n1hp, n2p, overlaph->size(), smooth2->size(), 1));
	overlaphp.reset(new ZD11SparseMatrix(prob->A, 0));
	smooth2p.reset(new ZD11SparseMatrix(prob->H, 0));

	// Subselect the smoothing matrix
	for (auto ii = smooth2->begin(); ii != smooth2->end(); ++ii) {
		int row2p = i2_to_i2p(ii.row());
		int col2p = i2_to_i2p(ii.col());
		smooth2p->set(row2p, col2p, ii.val());
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
		GridCell const &gc1((*grid1)[i1]);

		native_area1hp[i1hp] = overlap_area1hp[i1hp] * (gc1.native_area / gc1.proj_area);
	}

	// Set up GridCell pointers for grid2p
	grid2p.clear();
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GridCell const &gc2((*grid2)[i2 + grid2->index_base]);
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

	// Get initial approximation of answer (that satisfies our constraints)
	double * const Z2p(prob->X);
	overlaphp->multiplyT(&Z1hp[0], Z2p);
	for (int i2p=0; i2p < n2p; ++i2p) {
		GridCell const &gc2(*grid2p[i2p]);
		Z2p[i2p] *= gc2.proj_area / (overlap_area2p[i2p] * gc2.native_area);
	}

	// Re-arrange mass within each GCM gridcell (grid1)
	if (use_snowdrift) {
		// RHS of equality constraints
		for (int i1hp=0; i1hp<n1hp; ++i1hp) {
			double val = Z1hp[i1hp] * overlap_area1hp[i1hp];

			// We have "-val" here because the constraint is phrased as "Ax + c = 0"
			// See the GALAHAD documentation eqp.pdf
			prob->C[i1hp] = -val;
			// prob->C_l[i1hp] = -val;
			// prob->C_u[i1hp] = -val;
		}

		// Solve the QP problem
		ret = eqp_solve_simple_(prob->main);
	}

//	store_result(_i2p_to_i2, Z2, Z2p,
//		&overlap_area2p[0], &grid2p[0], merge_or_replace);

	// Merge results into Z2
	for (int i2p=0; i2p < n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		double X0 = Z2(i2);		// Original value

		GridCell const &gc2(*grid2p[i2p]);
		double X1 = Z2p[i2p] * (gc2.proj_area / gc2.native_area);	// Regrid result

		switch(merge_or_replace) {
			case MergeOrReplace::MERGE : {
				double overlap_fraction = overlap_area2p[i2p] / gc2.proj_area;
				Z2[i2] =
					(1.0 - overlap_fraction) * Z2[i2] +		// Original value
					overlap_fraction * X1;					// Our new value
			} break;
			default :		// REPLACE
				Z2(i2) = X1;
			break;
		}
	}

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
	// Select out Z, and scale to projected space
	std::vector<double> Z2p;
	Z2p.reserve(n2p);
	for (int i2p=0; i2p<n2p; ++i2p) {
		int i2 = i2p_to_i2(i2p);
		GridCell const &gc2(*grid2p[i2p]);
		Z2p.push_back(Z2(i2) * (gc2.native_area / gc2.proj_area));
	}

	// Regrid by multiplying by overlap matrix (HNTR regridding)
	double Z1hp[n1hp];
	overlaphp->multiply(&Z2p[0], Z1hp);

	// Scale back to Z/m^2
	for (int i1hp=0; i1hp < n1hp; ++i1hp) {
		// Z1hp[i1hp] *= proj_area1hp[i1hp] / (overlap_area1hp[i1hp] * native_area1hp[i1hp]);
		Z1hp[i1hp] /= native_area1hp[i1hp];		// Because proj_area1hp == overlap_area1hp
	}

	// Merge results into Z1
	for (int i1hp=0; i1hp < n1hp; ++i1hp) {
		int i1h = i1hp_to_i1h(i1hp);
		int i1 = i1h_to_i1(i1h);
		int hclass = get_hclass(i1h, i1);
		double X0 = Z1[hclass](i1);

		double X1 = Z1hp[i1hp] * (proj_area1hp[i1hp] / native_area1hp[i1hp]);	// Regrid result

		// Merge makes no sense here, because by definition,
		// proj_area1hp == overlap_area1hp.  So merge just reduces
		// to replace
		Z1[hclass][i1] = X1;
	}
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
