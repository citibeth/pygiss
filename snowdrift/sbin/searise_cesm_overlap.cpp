#include <boost/bind.hpp>
#include <cstdlib>
#include "Grid_LatLon.hpp"
#include "Grid_XY.hpp"
#include "OverlapMatrix.hpp"
#include "clippers.hpp"
#include "constants.hpp"

static const double km = 1000.0;

using namespace giss;

int main(int argc, char **argv)
{
#if 1
	char *str = getenv("RPFISCHE_DATA");
	if (!str) str = ".";
	std::string data_root(str);
	std::string cesm_fname = data_root + "/cesm/griddata_0.9x1.25_USGS_070110.nc";
#else
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <CESM-gridfile>\n", argv[0]);
		exit(-1);
	}
	std::string cesm_fname(argv[1]);
#endif

	// ------------- Set up the local ice grid
	printf("// ------------- Set up the local ice grid\n");

#if 1
	// The true exact SeaRISE grid
	std::unique_ptr<Grid_XY> ice_grid = Grid_XY::new_grid("ice",
		(- 800.0 - 2.5)*km, (- 800.0 + 300.0*5 + 2.5)*km,   5*km,
		(-3400.0 - 2.5)*km, (-3400.0 + 560.0*5 + 2.5)*km,   5*km,
		boost::bind(&EuclidianClip::keep_all, _1));
#else
	// Approximate SeaRISE grid
	std::unique_ptr<Grid_XY> ice_grid = Grid_XY::new_grid("ice",
		(-800)*km, (-800 + 300*5)*km,     5*km,
		(-3400)*km, (-3400 + 560*5)*km,   5*km,
		boost::bind(&EuclidianClip::keep_all, _1));
#endif

	printf("Ice grid has %ld cells\n", ice_grid->size());

	// ------------- Set up the projection (Same as SeaRISE)
	fflush(stdout);
	printf("// ------------- Set up the projection\n");
	double proj_lon_0 = -39;
	double proj_lat_0 = 90;
	char sproj[100];
	sprintf(sproj,
		"+proj=stere +lon_0=%f +lat_0=%f +lat_ts=71.0 +ellps=WGS84",
		proj_lon_0, proj_lat_0);

	// ------------- Set up the GCM Grid (read from CESM)
	fflush(stdout);
	printf("// ------------- Set up the GCM Grid\n");
	printf("Reading CESM file: %s\n", cesm_fname.c_str());
	// int points_in_side = 10;
	int points_in_side = 2;
	std::unique_ptr<Grid_LatLon> gcm_grid = Grid_LatLon::read_cesm(
		"gcm", Proj(sproj), points_in_side, cesm_fname,
		boost::bind(&SphericalClip::azimuthal,
			proj_lon_0, proj_lat_0, 40, _1, _2, _3, _4),
		boost::bind(&EuclidianClip::poly, ice_grid->bounding_box(), _1));
	printf("GCM grid has %ld cells\n", gcm_grid->size());

	// ------------- Compute the Overlap Matrix
	fflush(stdout);
	printf("// ------------- Compute the Overlap Matrix\n");
	OverlapMatrix overlap;
	overlap.set_grids(&*gcm_grid, &*ice_grid);


	// ------------- Write it out to NetCDF
	fflush(stdout);
	printf("// ------------- Write it out to NetCDF\n");
	std::string arg0(argv[0]);
	overlap.to_netcdf(arg0 + ".nc");
}
