#include <boost/bind.hpp>
#include "Grid_LatLon.hpp"
#include "constants.hpp"
#include "geodesy.hpp"
#include "clippers.hpp"
#include "maputils.hpp"

namespace giss {

template <typename T>
inline int sgn(T val) {
    return (val > T(0)) - (val < T(0));
}

// ---------------------------------------------------------

/** Computes the exact surface area of a lat-lon grid box on the
surface of the sphere.
NOTE: lon0 must be numerically less than lon1.
NOTE: lat0 and lat1 cannot cross the equator. */
inline double graticule_area_exact(double lat0_deg, double lat1_deg,
	double lon0_deg, double lon1_deg)
{
	double delta_lon_deg = lon1_deg - lon0_deg;
	delta_lon_deg = loncorrect(delta_lon_deg, 0);

//printf("delta_lon_deg=%f\n", delta_lon_deg);

	double lat0 = lat0_deg * D2R;
	double lat1 = lat1_deg * D2R;
	double delta_lon = delta_lon_deg * D2R;

//printf("lat1=%f, abs(lat1)=%f\n",lat1,std::abs(lat1));
	return delta_lon * (EQ_RAD*EQ_RAD) * (sin(lat1) - sin(lat0));
}

/** The polar graticule is a (latitudinal) circle centered on the pole.
This computes its area. */
inline double polar_graticule_area_exact(double radius_deg)
{

	// See http://en.wikipedia.org/wiki/Spherical_cap

	double theta = radius_deg * D2R;
	return 2.0 * M_PI * (EQ_RAD * EQ_RAD) * (1.0 - cos(theta));
}

// ---------------------------------------------------------
// ---------------------------------------------------------
/** 
@param lonb Longitude boundaries in the theoretical grid
@param south_pole Include the south pole "skullcap" in the theoretical grid?
       Must be false of lonb[0] == -90
*/
void Grid_LatLon::init(
std::vector<double> const &lonb,
std::vector<double> const &latb,
bool const south_pole, bool const north_pole,
Proj &&_proj,
int _points_in_side,
boost::function<bool(double, double, double, double)> const &spherical_clip,
boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip)
{
	Grid_LatLon *grid = this;
	grid->points_in_side = _points_in_side;
	grid->proj = std::move(_proj);
	printf("Using projection: \"%s\"\n", proj.get_def().c_str());

	// Get sphere/ellipsoid we're operating on
	Proj llproj(proj.latlong_from_proj());
	printf("Using lat-lon projection: \"%s\"\n", llproj.get_def().c_str());


	if (south_pole && latb[0] == -90.0) {
		std::cerr << "latb[] cannot include -90.0 if you're including the south pole cap" << std::endl;
		throw std::exception();
	}
	if (south_pole && latb.back() == 90.0) {
		std::cerr << "latb[] cannot include 90.0 if you're including the north pole cap" << std::endl;
		throw std::exception();
	}

//	std::unique_ptr<Grid_LatLon> grid(new Grid_LatLon());
	grid->south_pole = south_pole;
	grid->north_pole = north_pole;
	grid->lon_boundaries = lonb;
	grid->lat_boundaries = latb;


	// ------------------- Set up the GCM Grid
	const int south_pole_offset = (south_pole ? 1 : 0);
	const int north_pole_offset = (north_pole ? 1 : 0);

	int IM = lonb.size() - 1;
	int JM = latb.size() - 1 + south_pole_offset + north_pole_offset;

	// Get a bunch of points.  (i,j) is gridcell's index in canonical grid
	gc::Polygon_2 poly;
	for (int ilat=0; ilat < latb.size()-1; ++ilat) {
		double lat0 = latb[ilat];
		double lat1 = latb[ilat+1];

		for (int ilon=0; ilon< lonb.size()-1; ++ilon) {
			poly.clear();
			double lon0 = lonb[ilon];
			double lon1 = lonb[ilon+1];

//printf("(ilon, ilat) = (%d, %d)\n", lon0, lat0);
//printf("values = %f %f %f %f\n", lon0, lat0, lon1, lat1);

			if (!spherical_clip(lon0, lat0, lon1, lat1)) continue;

			// Project the grid cell boundary to a planar polygon
			ll2xy_latitude(llproj, proj, poly, points_in_side, lon0,lon1, lat0);
			ll2xy_meridian(llproj, proj, poly, points_in_side, lon1, lat0,lat1);
			ll2xy_latitude(llproj, proj, poly, points_in_side, lon1,lon0, lat1);
			ll2xy_meridian(llproj, proj, poly, points_in_side, lon0, lat1,lat0);

			if (!euclidian_clip(poly)) continue;

			// Figure out how to number this grid cell
			int j_base0 = ilat + south_pole_offset;	// 0-based 2-D index
			int i_base0 = ilon;						// 0-based 2-D index
			int index = (j_base0 * IM + i_base0) + index_base;
			grid->add_cell(GridCell(poly, index,
				graticule_area_exact(lat0,lat1,lon0,lon1)));
		}
	}

	// Make the polar caps (if this grid specifies them)

	// North Pole cap
	double lat = latb.back();
	if (north_pole && spherical_clip(0, lat, 360, 90)) {
		gc::Polygon_2 pole;
		for (int ilon=0; ilon< lonb.size()-1; ++ilon) {
			double lon0 = lonb[ilon];
			double lon1 = lonb[ilon+1];
			ll2xy_latitude(llproj, proj, pole, points_in_side, lon0,lon1, lat);
		}
		if (euclidian_clip(pole)) grid->add_cell(
			GridCell(pole, IM*JM - 1 + index_base,
			polar_graticule_area_exact(90.0 - lat)));
	}

	// South Pole cap
	lat = latb[0];
	if (south_pole && spherical_clip(0, -90, 360, lat)) {
		gc::Polygon_2 pole;
		for (int ilon=lonb.size()-1; ilon >= 1; --ilon) {
			double lon0 = lonb[ilon];		// Make the circle counter-clockwise
			double lon1 = lonb[ilon-1];
			ll2xy_latitude(llproj, proj, pole, points_in_side, lon0,lon1, lat);
		}
		if (euclidian_clip(pole)) grid->add_cell(
			GridCell(pole, 0 + index_base, polar_graticule_area_exact(90.0 + lat)));
	}
}

// ---------------------------------------------------------

static void Grid_LatLon_netcdf_write(
	boost::function<void()> const &parent,
	NcFile *nc, Grid_LatLon const *grid, std::string const &generic_name)
{
	parent();

	NcVar *lonbVar = nc->get_var((generic_name + ".lon_boundaries").c_str());
	NcVar *latbVar = nc->get_var((generic_name + ".lat_boundaries").c_str());

	lonbVar->put(&grid->lon_boundaries[0], grid->lon_boundaries.size());
	latbVar->put(&grid->lat_boundaries[0], grid->lat_boundaries.size());

#if 0
	NcVar *poleVar = nc->get_var((generic_name + ".south_pole").c_str());
	int ipole = grid->south_pole;
	poleVar->put(&ipole, 1);

	poleVar = nc->get_var((generic_name + ".north_pole").c_str());
	ipole = grid->north_pole;
	poleVar->put(&ipole, 1);
#endif

}

boost::function<void ()> Grid_LatLon::netcdf_define(NcFile &nc, std::string const &generic_name) const
{
	auto parent = Grid::netcdf_define(nc, generic_name);

//	NcDim *oneDim = nc.get_dim("one");
//	nc.add_var((generic_name + ".south_pole").c_str(), ncInt, oneDim);
//	nc.add_var((generic_name + ".north_pole").c_str(), ncInt, oneDim);

	NcDim *lonbDim = nc.add_dim((generic_name + ".lon_boundaries.length").c_str(),
		this->lon_boundaries.size());
	NcVar *lonbVar = nc.add_var((generic_name + ".lon_boundaries").c_str(),
		ncDouble, lonbDim);

	NcDim *latbDim = nc.add_dim((generic_name + ".lat_boundaries.length").c_str(),
		this->lat_boundaries.size());
	NcVar *latbVar = nc.add_var((generic_name + ".lat_boundaries").c_str(),
		ncDouble, latbDim);

	NcVar *infoVar = nc.get_var((generic_name + ".info").c_str());
	infoVar->add_att("north_pole_cap", north_pole ? 1 : 0);
	infoVar->add_att("south_pole_cap", south_pole ? 1 : 0);
	infoVar->add_att("points_in_side", points_in_side);
	infoVar->add_att("projection", proj.get_def().c_str());
	infoVar->add_att("latlon_projection", proj.latlong_from_proj().get_def().c_str());


	return boost::bind(&Grid_LatLon_netcdf_write, parent, &nc, this, generic_name);
}


// ---------------------------------------------------------
// Latitude and longitude gridcell boundaries for the 4x5 grid

const std::vector<double> lonb_4x5 = {-180,-175,-170,-165,-160,-155,-150,-145,-140,-135,-130,-125,-120,-115,-110,-105,-100,-95,-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180};

const std::vector<double> latb_4x5 = {-88,-84,-80,-76,-72,-68,-64,-60,-56,-52,-48,-44,-40,-36,-32,-28,-24,-20,-16,-12,-8,-4,0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88};

std::unique_ptr<Grid_LatLon> Grid_LatLon::new_grid_4x5(
std::string const &name,
Proj &&proj,
int points_in_side,
boost::function<bool(double, double, double, double)> const &spherical_clip,
boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip)
{
	int south_pole_offset = 1;
	int north_pole_offset = 1;
	int IM = lonb_4x5.size() - 1;
	int JM = latb_4x5.size() - 1 + south_pole_offset + north_pole_offset;
	int max_index = IM*JM;
	std::unique_ptr<Grid_LatLon> grid(new Grid_LatLon(name, 1, max_index));
	grid->init(
		lonb_4x5, latb_4x5, true, true,
		std::move(proj), points_in_side,
		spherical_clip, euclidian_clip);
	return grid;
}

// ---------------------------------------------------------
// Latitude and longitude gridcell boundaries for the 2x2.5 grid

std::unique_ptr<Grid_LatLon> Grid_LatLon::new_grid_2x2_5(
std::string const &name,
Proj &&proj,
int points_in_side,
boost::function<bool(double, double, double, double)> const &spherical_clip,
boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip)
{
	// Create the 2x2.5 grid from the 4x5 grid.
	std::vector<double> lonb_2x2_5;
	for (int i=0; i<lonb_4x5.size()-1; ++i) {
		lonb_2x2_5.push_back(lonb_4x5[i]);
		lonb_2x2_5.push_back(.5*(lonb_4x5[i] + lonb_4x5[i+1]));
	}
	lonb_2x2_5.push_back(lonb_4x5.back());

	std::vector<double> latb_2x2_5;
	for (int i=0; i<latb_4x5.size()-1; ++i) {
		latb_2x2_5.push_back(latb_4x5[i]);
		latb_2x2_5.push_back(.5*(latb_4x5[i] + latb_4x5[i+1]));
	}
	latb_2x2_5.push_back(latb_4x5.back());

//for (double lon : lonb_2x2_5) printf("%f\n", lon);

	int south_pole_offset = 1;
	int north_pole_offset = 1;
	int IM = lonb_2x2_5.size() - 1;
	int JM = latb_2x2_5.size() - 1 + south_pole_offset + north_pole_offset;
	int max_index = IM*JM;
	std::unique_ptr<Grid_LatLon> grid(new Grid_LatLon(name, 1, max_index));
	grid->init(
		lonb_2x2_5, latb_2x2_5, true, true,
		std::move(proj), points_in_side,
		spherical_clip, euclidian_clip);
	return grid;
}

void Grid_LatLon::read_from_netcdf(NcFile &nc, std::string const &vname)
{
	Grid::read_from_netcdf(nc, vname);

	lon_boundaries = read_double_vector(nc, vname + ".lon_boundaries");
	lat_boundaries = read_double_vector(nc, vname + ".lat_boundaries");

	// ... don't bother reading the rest of stuff for now...
	points_in_side = -1;
	south_pole = false;
	north_pole = false;
	// proj = ...;
}


}
