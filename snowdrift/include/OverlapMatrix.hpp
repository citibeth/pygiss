#pragma once

#include <netcdfcpp.h>
#include "Grid.hpp"
#include "RTree.h"

/// Should we use Rtrees to speed up overlap matrix computation?
#define USE_RTREE 1

/// Should we be paranoid in checking things in our overlap matrix?
#define PARANOID_OVERLAP 1


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

class OverlapMatrix_NC_Overlap;

class OverlapMatrix {

	typedef RTree<GridCell const *, double, 2, double> MyRTree;

	Grid const *grid1;		// One of the two grids
	MyRTree rtree;

	/** Index of the grid cells that have actually
	overlapped in the two grids. */
	std::set<UsedGridCell> used[2];

	std::vector<GridCellOverlap> overlaps;

public :
	void set_grid1(Grid const *grid1);

	/** Computes the overlap matrix! */
	void add_all(Grid const &grid0);

	/** Computes one row of overlap matrix --- useful if grid0 is too large
	to store in memory all at once. */
	void add_row(GridCell const &gc0);

	bool add_pair(GridCell const *gc0, GridCell const *gc1);

	// =============================================================
	// NetCDF Stuff...

	std::unique_ptr<OverlapMatrix_NC_Overlap> netcdf_define(
		NcFile &nc);

//	static void netcdf_write(OverlapMatrix_NC_Overlap &);


};	// class OverlapMatrix


// =============================================================
	struct OverlapMatrix_NC_Used {	// A little closure
		NcFile &nc;
		std::set<UsedGridCell> const &used;

		NcDim *lenDim;
		NcVar *indexVar;
		NcVar *native_areaVar;
		NcVar *proj_areaVar;

		// Define variables in NetCDF
		OverlapMatrix_NC_Used(NcFile &_nc, std::set<UsedGridCell> &_used, std::string const &name);

		// Write data to NetCDF
		void write();

	};

	struct OverlapMatrix_NC_Overlap { // closure
		std::vector<GridCellOverlap> const *overlaps;

		std::unique_ptr<OverlapMatrix_NC_Used> nc_used[2];

		NcDim *lenDim;
		NcDim *num_gridsDim;
		NcVar *grid_indexVar;
		NcVar *areaVar;

		void write();
	};






} 	// namespace giss

