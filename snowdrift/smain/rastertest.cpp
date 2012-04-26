#include <boost/bind.hpp>

#include "Grid_LatLon.hpp"
#include "Grid_XY.hpp"
#include "OverlapMatrix.hpp"
#include "clippers.hpp"

static const double km = 1000.0;

using namespace giss;

int main(int argc, char **argv)
{
	std::unique_ptr<Grid> grid = Grid::from_netcdf("xy_overlap.nc", "grid1");
}

