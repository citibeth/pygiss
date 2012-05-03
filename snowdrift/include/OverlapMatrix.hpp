#pragma once

#include <netcdfcpp.h>
#include "Grid.hpp"
#include "RTree.h"


namespace giss {

/** Used to generate the overlap matrix, not a data structure to read it back in. */
class OverlapMatrix {

	Grid *grid1, *grid2;

	std::unique_ptr<VectorSparseMatrix> overlaps;

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

	static std::unique_ptr<OverlapMatrix> from_netcdf(std::string const &fname);

};	// class OverlapMatrix



} 	// namespace giss

