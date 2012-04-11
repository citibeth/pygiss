#pragma once

#include "geometry.hpp"
#include "Proj.hpp"

namespace giss {


	/** Project a latitude line segment
	n >=1, number of points to add to polygoon.  Don't add (lat,lon1). */
	void ll2xy_latitude(Proj const &llproj, Proj const &proj, gc::Polygon_2 &poly, int n,
		double lon0, double lon1, double lat);


	/** Project a latitude line segment
	n >=1, number of points to add to polygoon.  Don't add (lat,lon1). */
	void ll2xy_meridian(Proj const &llproj, Proj const &proj, gc::Polygon_2 &poly, int n,
		double lon, double lat0, double lat1);




// Normalises a value of longitude to the range starting at min degrees.
// @return The normalised value of longitude.
inline double loncorrect(double lon, double min)
{
    double max = min + 360.0;

	while (lon >= max) lon -= 360.0;
	while (lon < min) lon += 360.0;

	return lon;
}


}	// namespace giss
