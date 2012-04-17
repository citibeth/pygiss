#pragma once

#include <netcdfcpp.h>
#include "Grid.hpp"
#include "RTree.h"


namespace giss {

/** Describes two grid cells that overlap */
struct GridCellOverlap {
	int cell_index[2];	// The cell index of gridcell in each grid that overlaps
	double area;	// Area of overlap

	GridCellOverlap(int _index0, int _index1, double _area) :
		cell_index({_index0, _index1}), area(_area) {}

	/** Sort overlap matrix in row major form, common for sparse matrix representations */
	bool operator<(GridCellOverlap const &b) const
	{
		if (cell_index[0] < b.cell_index[0]) return true;
		if (cell_index[0] > b.cell_index[0]) return false;
		return (cell_index[1] < b.cell_index[1]);
	}

};


/** Describes one grid cell from a grid that participates in the overlap */
class UsedGridCell {
public:
	int index;
	double native_area;
	double proj_area;

	UsedGridCell(GridCell const &gc) :
		index(gc.index), native_area(gc.native_area), proj_area(gc.proj_area) {}

	bool operator<(UsedGridCell const &b) const
		{ return index < b.index; }
};

class OverlapMatrix {

	Grid *grid1, *grid2;

	/** Index of the grid cells that have actually
	overlapped in the two grids. */
	std::set<UsedGridCell> used[2];

	std::vector<GridCellOverlap> overlaps;

public :
	void set_grids(Grid *grid1, Grid *grid2);


	bool add_pair(GridCell const *gc0, GridCell const *gc1);

	// =============================================================
	// NetCDF Stuff...

private:

	void netcdf_write(NcFile *nc,
		boost::function<void ()> const &nc_used_write0,
		boost::function<void ()> const &nc_used_write1);

public:
	boost::function<void ()> netcdf_define(NcFile &nc);

	/** Easy all-in-one function to write this out to a netCDF File */
	void to_netcdf(std::string const &fname);

};	// class OverlapMatrix



} 	// namespace giss

