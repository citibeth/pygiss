#include <boost/bind.hpp>
#include "OverlapMatrix.hpp"
#include <algorithm>

namespace giss {

inline void add_used(std::map<int, Overlap> &used, int index0, double area)
{
	auto f0 = used.find(index0);
	if (f0 == used.end()) {
		used.insert(index0, Overlap(area));
	} else {
		f0->total_coverage += area;
	}
}


void OverlapMatrix::add_overlap(int index0, int index1, double area)
{
	add_used(used[0], index0, area);
	add_used(used[1], index1, area);
	overlaps.push_back(SparseCell(index0, index1, area));
}

/** Always returns true */
bool OverlapMatrix::add_pair(GridCell const *gc0, GridCell const *gc1)
{

	// Adding this check increases time requirements :-(
	// if (!CGAL::do_intersect(gc0->poly, gc1->poly)) return true;

	// Now compute the overlap area
	double area = CGAL::to_double(overlap_area(gc0->poly, gc1->poly));
	if (area == 0) return true;

	add_overlap(gc0->index, gc1->index, area);

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
			printf("Processed %d of %d from grid1, total overlaps = %d\n", i+1, grid1->cells().size(), overlaps.size());
		}

	}

	// Sort the matrix
	std::sort(overlaps.begin(), overlaps.end());


}

// -----------------------------------------------------------
// ===========================================================
// NetCDF Stuff


void nc_used_netcdf_write(
	NcFile *nc, std::set<UsedGridCell> *used, std::string const name)
{
	NcVar *indexVar = nc->get_var((name + ".overlap_cells").c_str());

	int i=0;
	for (auto ugc = used->begin(); ugc != used->end(); ++ugc) {
		indexVar->set_cur(i);
		indexVar->put(&ugc->index, 1);

		++i;
	}
}

boost::function<void ()> nc_used_netcdf_define(
	NcFile &nc, std::set<UsedGridCell> *used, std::string const &name)
{
	auto lenDim = nc.add_dim((name + ".num_overlap_cells").c_str(), used->size());
	auto var = nc.add_var((name + ".overlap_cells").c_str(), ncInt, lenDim);
	var->add_att("description",
		"Index of each grid cell that participates in overlap with"
		" the other grid.  Subset of realized_cells");

//	nc.add_var((name + ".grid_index").c_str(), ncInt, lenDim);
//	nc.add_var((name + ".native_area").c_str(), ncDouble, lenDim);
//	nc.add_var((name + ".proj_area").c_str(), ncDouble, lenDim);

	return boost::bind(&nc_used_netcdf_write, &nc, used, name);
}

// -------------------------------------------------------------
void OverlapMatrix::netcdf_write(NcFile *nc,
	boost::function<void ()> const &nc_used_write0,
	boost::function<void ()> const &nc_used_write1)
{
	nc_used_write0();
	nc_used_write1();

	NcVar *grid_indexVar = nc->get_var("overlap.overlap_cells");
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
	auto grid_indexVar = nc.add_var("overlap.overlap_cells", ncInt, lenDim, num_gridsDim);
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

std::unique_ptr<OverlapMatrix> OverlapMatrix::from_netcdf(
Grid *grid1, Grid *grid2, NcFile &nc)
{
	std::unique_ptr<OverlapMatrix> ret(new OverlapMatrix());
	ret->grid1 = grid1;
	ret->grid2 = grid2;

//	std::unique_ptr<Grid> grid1(Grid::from_netcdf(nc, "grid1"));
//	std::unique_ptr<Grid> grid2(Grid::from_netcdf(nc, "grid2"));

	NcVar *vcells = nc.get_var("overlap.overlap_cells".c_str());
	NcVar *varea = nc.get_var("overlap.area".c_str());
	long noverlap = varea->get_dim(0)->size();

	std::vector<double> area(noverlap);
	std::vector<int> cells(noverlap*2);

	ret->overlaps.reserve(noverlap);
	for (int i=0; i<noverlap; ++i) {
		ret->used[0].insert(cells[i*2 + 0]);
		ret->used[1].insert(cells[i*2 + 1]);
		ret->overlaps.insert(SparseCell(
			cells[i*2 + 0], cells[i*2 + 1], area[i]);
	}
}




} 	// namespace giss
