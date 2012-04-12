#include <boost/bind.hpp>
#include "OverlapMatrix.hpp"
#include "RTree.h"
#include <algorithm>

namespace giss {


// ---------------------------------------------------------
void OverlapMatrix::add_all(Grid const *_grid1)
{
	grid1 = _grid1;
	int i=0;
	for (auto ii0=grid1->cells().begin(); ii0 != grid1->cells().end(); ++ii0) {
		GridCell const &gc0 = ii0->second;


//		if (overlaps.size() % 1000 == 1)
//			printf("%d overlaps, Looking at index=%d (%d of %d) on grid 0\n",
//				overlaps.size(), gc0.index, i+1, grid1->cells().size());
//		std::cout << "i0=" << gc0.index << "   gc0 = " << gc0.poly << std::endl;
//		std::cout << "gc0 bounding box = " << gc0.bounding_box << std::endl;

		add_row(gc0);

		// Logging
		++i;
		if (i % 1000 == 0) {
			printf("Processed %d of %d from grid1, total overlaps = %d\n", i+1, grid1->cells().size(), overlaps.size());
		}

	}

	// Sort the matrix
	std::sort(overlaps.begin(), overlaps.end());
}



/** Always returns true */
bool OverlapMatrix::add_pair(GridCell const *gc0_p, GridCell const *gc1_p)
{
	// Rename vars
	GridCell const &gc0(*gc0_p);
	GridCell const &gc1(*gc1_p);

#if !USE_RTREE
	// Quick test to see if they might overlap
	// (This is not needed in the RTree version)
	if (!CGAL::do_intersect(gc0.bounding_box, gc1.bounding_box)) {
//		printf("Unexpected bounding box intersection!\n");
		return true;
	}
#endif


	// Adding this check increases time requirements :-(
	// if (!CGAL::do_intersect(gc0.poly, gc1.poly)) return true;

	// Now compute the overlap area
	double area = CGAL::to_double(overlap_area(gc0.poly, gc1.poly));
	if (area == 0) return true;

	used[0].insert(UsedGridCell(gc0));
	used[1].insert(UsedGridCell(gc1));
	overlaps.push_back(GridCellOverlap(gc0.index, gc1.index, area));

#if 0
if (!(		// If gc1 is NOT fully contained in gc0 (in bounding box)
gc0.bounding_box.xmin() < gc1.bounding_box.xmin() && gc0.bounding_box.xmax() > gc1.bounding_box.xmax() &&
gc0.bounding_box.ymin() < gc1.bounding_box.ymin() && gc0.bounding_box.ymax() > gc1.bounding_box.ymax())) {
	std::cout << "gc1 not contained in gc0: " << gc0.bounding_box << " **** " << gc1.bounding_box << std::endl;
}
#endif

	return true;
}


#if USE_RTREE

// This REALLY speeds things up a lot
// New Version: use RTrees
/** Creates an RTree out of grid1 */
void OverlapMatrix::set_grid2(Grid const *_grid2)
{
	grid2 = _grid2;

	rtree.RemoveAll();
	double min[2];
	double max[2];
	for (auto ii1=grid2->cells().begin(); ii1 != grid2->cells().end(); ++ii1) {
		GridCell const &gc1 = ii1->second;

		min[0] = CGAL::to_double(gc1.bounding_box.xmin());
		min[1] = CGAL::to_double(gc1.bounding_box.ymin());
		max[0] = CGAL::to_double(gc1.bounding_box.xmax());
		max[1] = CGAL::to_double(gc1.bounding_box.ymax());

		// Deal with floating point...
		const double eps = 1e-7;
		double epsilon_x = eps * std::abs(max[0] - min[0]);
		double epsilon_y = eps * std::abs(max[1] - min[1]);
		min[0] -= epsilon_x;
		min[1] -= epsilon_y;
		max[0] += epsilon_x;
		max[1] += epsilon_y;

//std::cout << gc1.poly << std::endl;
//std::cout << gc1.bounding_box << std::endl;
//printf("(%g,%g) -> (%g,%g)\n", min[0], min[1], max[0], max[1]);
		rtree.Insert(min, max, &gc1);
	}
}

/** Adds a contribution from everything in grid2 that intersects with gc0 */
void OverlapMatrix::add_row(GridCell const &gc0)
{
	double min[2];
	double max[2];

	min[0] = CGAL::to_double(gc0.bounding_box.xmin());
	min[1] = CGAL::to_double(gc0.bounding_box.ymin());
	max[0] = CGAL::to_double(gc0.bounding_box.xmax());
	max[1] = CGAL::to_double(gc0.bounding_box.ymax());

	int nfound = rtree.Search(min, max,
		boost::bind(&OverlapMatrix::add_pair, this, &gc0, _1));
//	printf("RTree.Search() found %d entries, %d called, %d added\n", nfound, add_called, added_in_row);
}

#else	// !USE_RTREE
// Old version: n^2 search
void OverlapMatrix::set_grid2(Grid const *_grid2)
{
	this->grid2 = _grid2;
}

/** Adds a contribution from everything in grid2 that intersects with gc0 */
void OverlapMatrix::add_row(GridCell const &gc0)
{
	for (auto ii1=grid2->cells().begin(); ii1 != grid2->cells().end(); ++ii1) {
		GridCell const &gc1 = ii1->second;
		add_pair(&gc0, &gc1);
	}
}
#endif
// -----------------------------------------------------------
// ===========================================================
// NetCDF Stuff


void nc_used_netcdf_write(
	NcFile *nc, std::set<UsedGridCell> *used, std::string const name)
{
	NcVar *indexVar = nc->get_var((name + ".grid_index").c_str());
	NcVar *native_areaVar = nc->get_var((name + ".native_area").c_str());
	NcVar *proj_areaVar = nc->get_var((name + ".proj_area").c_str());

	int i=0;
	for (auto ugc = used->begin(); ugc != used->end(); ++ugc) {
		indexVar->set_cur(i);
		indexVar->put(&ugc->index, 1);
		native_areaVar->set_cur(i);
		native_areaVar->put(&ugc->native_area, 1);
		proj_areaVar->set_cur(i);
		proj_areaVar->put(&ugc->proj_area, 1);

		++i;
	}
}

boost::function<void ()> nc_used_netcdf_define(
	NcFile &nc, std::set<UsedGridCell> *used, std::string const &name)
{
	auto lenDim = nc.add_dim((name + ".overlap_grid_cells").c_str(), used->size());
	nc.add_var((name + ".grid_index").c_str(), ncInt, lenDim);
	nc.add_var((name + ".native_area").c_str(), ncDouble, lenDim);
	nc.add_var((name + ".proj_area").c_str(), ncDouble, lenDim);

	return boost::bind(&nc_used_netcdf_write, &nc, used, name);
}

// -------------------------------------------------------------
void OverlapMatrix::netcdf_write(NcFile *nc,
	boost::function<void ()> const &nc_used_write0,
	boost::function<void ()> const &nc_used_write1)
{
	nc_used_write0();
	nc_used_write1();

	NcVar *grid_indexVar = nc->get_var("overlap.grid_index");
	NcVar *areaVar = nc->get_var("overlap.area");

	int i=0;
	for (auto ov = overlaps.begin(); ov != overlaps.end(); ++ov) {
		grid_indexVar->set_cur(i,0);
		grid_indexVar->put(ov->cell_index, 1,2);

		areaVar->set_cur(i);
		areaVar->put(&ov->area, 1);

		++i;
	}
}

boost::function<void ()> OverlapMatrix::netcdf_define(
NcFile &nc)
{
	boost::function<void ()> nc_used_write0 =
		nc_used_netcdf_define(nc, &used[0], "grid1");
	boost::function<void ()> nc_used_write1 =
		nc_used_netcdf_define(nc, &used[1], "grid2");

	auto lenDim = nc.add_dim("overlap.num_overlaps", overlaps.size());
	auto num_gridsDim = nc.add_dim("overlap.num_grids", 2);
	auto grid_indexVar = nc.add_var("overlap.grid_index", ncInt, lenDim, num_gridsDim);
	auto areaVar = nc.add_var("overlap.area", ncDouble, lenDim);

	return boost::bind(&OverlapMatrix::netcdf_write, this, &nc,
		nc_used_write0,
		nc_used_write1);
}

void OverlapMatrix::to_netcdf(std::string const &fname)
{
	NcFile nc(fname.c_str(), NcFile::Replace);

	// Define stuff in NetCDF file
	auto ncd = netcdf_define(nc);
	auto gcmd = grid1->netcdf_define(nc, "grid1");
	auto iced = grid2->netcdf_define(nc, "grid2");

	// Write stuff in NetCDF file
	gcmd();
	iced();
	ncd();

	nc.close();
}


} 	// namespace giss
