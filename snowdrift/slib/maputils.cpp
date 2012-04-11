#include "maputils.hpp"
#include "geometry.hpp"
#include "Proj.hpp"
#include "constants.hpp"

namespace giss {

// ---------------------------------------------------------
/** Project a latitude line segment
n >=1, number of points to add to polygoon.  Don't add (lat,lon1). */
void ll2xy_latitude(Proj const &llproj, Proj const &proj, gc::Polygon_2 &poly, int n,
	double lon0, double lon1, double lat)
{

	for (int i=0; i<n; ++i) {
		double lon = lon0 + (lon1-lon0) * ((double)i/(double)n);
		double x,y;
		int err = transform(llproj, proj, lon*D2R, lat*D2R, x, y);
		poly.push_back(gc::Point_2(x,y));
	}
}

/** Project a latitude line segment
n >=1, number of points to add to polygoon.  Don't add (lat,lon1). */
void ll2xy_meridian(Proj const &llproj, Proj const &proj, gc::Polygon_2 &poly, int n,
	double lon, double lat0, double lat1)
{
	for (int i=0; i<n; ++i) {
		double lat = lat0 + (lat1-lat0) * ((double)i/(double)n);
		double x,y;
		transform(llproj, proj, lon*D2R, lat*D2R, x, y);
		// proj.ll2xy(lon, lat, x, y);
		poly.push_back(gc::Point_2(x,y));
	}
}

}
