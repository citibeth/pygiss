#include <boost/bind.hpp>
#include "OverlapMatrix.hpp"
#include <algorithm>

namespace giss {


/** Always returns true */
bool OverlapMatrix::add_pair(GridCell const *gc0, GridCell const *gc1)
{

	// Adding this check increases time requirements :-(
	// if (!CGAL::do_intersect(gc0->poly, gc1->poly)) return true;

	// Now compute the overlap area
	double area = CGAL::to_double(overlap_area(gc0->poly, gc1->poly));
	if (area == 0) return true;

	overlaps->set(gc0->index, gc1->index, area);

	return true;
}

// Used in a boost::bind
static bool add_pair_x(OverlapMatrix *om, GridCell const **gc1, GridCell const *gc2)
{
	return om->add_pair(*gc1, gc2);
}


void OverlapMatrix::set_grids(Grid *_grid1, Grid *_grid2)
{
	this->grid1 = _grid1;
	this->grid2 = _grid2;

	int n1 = grid1->max_index - grid1->index_base + 1;
	int n2 = grid2->max_index - grid2->index_base + 1;

	overlaps.reset(new VectorSparseMatrix(SparseDescr(
		n1, n2, 1,
		SparseMatrix::MatrixStructure::TRIANGULAR,
		SparseMatrix::TriangularType::LOWER)));


	int i=0;

	GridCell const *gc1;
	auto callback = boost::bind(&add_pair_x, this, &gc1, _1);
	Grid::RTree &rtree2(grid2->rtree());

	for (auto ii1=grid1->cells().begin(); ii1 != grid1->cells().end(); ++ii1) {
		gc1 = &ii1->second;

		double min[2];
		double max[2];

		min[0] = CGAL::to_double(gc1->bounding_box.xmin());
		min[1] = CGAL::to_double(gc1->bounding_box.ymin());
		max[0] = CGAL::to_double(gc1->bounding_box.xmax());
		max[1] = CGAL::to_double(gc1->bounding_box.ymax());

		int nfound = rtree2.Search(min, max, callback);

		// Logging
		++i;
		if (i % 1000 == 0) {
			printf("Processed %d of %d from grid1, total overlaps = %d\n", i+1, grid1->cells().size(), overlaps->size());
		}

	}

	// Sort the matrix
	overlaps->sort(SparseMatrix::SortOrder::ROW_MAJOR);
}

// -----------------------------------------------------------
// ===========================================================
// NetCDF Stuff


void OverlapMatrix::to_netcdf(std::string const &fname)
{
	NcFile nc(fname.c_str(), NcFile::Replace);

	// Define stuff in NetCDF file
	auto gcmd = grid1->netcdf_define(nc, "grid1");
	auto iced = grid2->netcdf_define(nc, "grid2");
	auto ncd = overlaps->netcdf_define(nc, "overlap");

	// Write stuff in NetCDF file
	gcmd();
	iced();
	ncd();

	nc.close();
}



} 	// namespace giss
