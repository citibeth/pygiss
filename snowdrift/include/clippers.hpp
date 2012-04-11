#pragma once

#include "geometry.hpp"

namespace giss {

class SphericalClip {
public:
	/** Clips everything too far from a central point */
	static bool azimuthal(
		double center_lon, double center_lat, double clip_distance_deg,
		double lon0, double lat0, double lon1, double lat1);



	/** Clips everything outside of a lat-lon box */
	static bool latlon(
		double min_lon, double min_lat, double max_lon, double max_lat,
		double lon0, double lat0, double lon1, double lat1);

	static bool keep_all(double lon0, double lat0, double lon1, double lat1);

};


class EuclidianClip {
public:

	/** @param clip_poly Only realize grid cells that intersect
	with this polygon (on the map) */
	static bool poly(gc::Polygon_2 const &clip_poly,
		gc::Polygon_2 const &grid_cell);

	static bool keep_all(gc::Polygon_2 const &grid_cell);

};

}
