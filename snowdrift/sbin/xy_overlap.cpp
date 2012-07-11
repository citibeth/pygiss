#include <boost/bind.hpp>

#include "Grid_LatLon.hpp"
#include "Grid_XY.hpp"
#include "OverlapMatrix.hpp"
#include "clippers.hpp"

static const double km = 1000.0;

using namespace giss;

int main(int argc, char **argv)
{
	// ------------- Set up the local ice grid
	printf("// ------------- Set up the local ice grid\n");
	std::unique_ptr<Grid_XY> ice_grid = Grid_XY::new_grid("ice",
		0*km, 1600*km,   50*km,
		0*km, 1600*km,   50*km,
		boost::bind(&EuclidianClip::keep_all, _1));

	// ------------- Set up the GCM Grid
	printf("// ------------- Set up the GCM Grid\n");
	std::unique_ptr<Grid_XY> gcm_grid = Grid_XY::new_grid("gcm",
		0*km, 1600*km,   200*km,
		0*km, 1600*km,   200*km,
		boost::bind(&EuclidianClip::poly, ice_grid->bounding_box(), _1));

	// ------------- Compute the Overlap Matrix
	printf("// ------------- Compute the Overlap Matrix\n");
	OverlapMatrix overlap;
	overlap.set_grids(&*gcm_grid, &*ice_grid);


	// ------------- Write it out to NetCDF
	fflush(stdout);
	printf("// ------------- Write it out to NetCDF\n");
	std::string arg0(argv[0]);
	overlap.to_netcdf(arg0 + ".nc");
}
