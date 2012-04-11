#pragma once

#include <boost/function.hpp>
#include "Proj.hpp"
#include "Grid.hpp"

namespace giss {

class Grid_LatLon : public Grid {
public:
	
	// Information about the full global grid (irrespective of overlap)
	std::vector<double> lon_boundaries;	// Cell boundaries
	std::vector<double> lat_boundaries;
	int points_in_side;					// # segments used to represent each side of a grid cell.
	Proj proj;							// Projection used to create this grid

	bool south_pole, north_pole;

	Grid_LatLon() : Grid("latlon") {}


	/**
	@param lonb x-axis boundaries of grid cells --- last one is 360 + first.
	@param latb y-axis boundaries of grid cells --- sorted from low to high
	*/
	void init(
		std::vector<double> const &lonb,
		std::vector<double> const &latb,
		bool const south_pole, bool const north_pole,
		Proj &&proj,
		int points_in_side,
		boost::function<bool(double, double, double, double)> const &spherical_clip,
		boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip);


	static std::unique_ptr<Grid_LatLon> new_grid_4x5(
		Proj &&proj,
		int points_in_side,
		boost::function<bool(double, double, double, double)> const &spherical_clip,
		boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip);

	static std::unique_ptr<Grid_LatLon> new_grid_2x2_5(
		Proj &&proj,
		int points_in_side,
		boost::function<bool(double, double, double, double)> const &spherical_clip,
		boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip);

	 virtual boost::function<void()> netcdf_define(NcFile &nc, std::string const &generic_name, std::string const &specific_name);

		
};


}
