#pragma once

#include <boost/function.hpp>
#include "Grid.hpp"
#include "geometry.hpp"

namespace giss {


class Grid_XY : public Grid {

public:

	// Information about the full grid (irrespective of overlap)
	std::vector<double> x_boundaries;	// Cell boundaries
	std::vector<double> y_boundaries;

//	std::vector<double> x_centers;	// Cell centers
//	std::vector<double> y_centers;

//	gc::Point_2


	Grid_XY(std::string const &name) : Grid("xy", name) {}

	/**
	@var xb x-axis boundaries of grid cells --- last one is 360 + first.
	@var yb y-axis boundaries of grid cells --- sorted from low to high
	@var clip_poly Only realize grid cells that intersect with this polygon (on the map)
	*/
	static std::unique_ptr<Grid_XY> new_grid(
		std::string const &name,
		std::vector<double> const &xb,
		std::vector<double> const &yb,
		boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip);

	static std::unique_ptr<Grid_XY> new_grid(
		std::string const &name,
		double x0, double x1, double dx,
		double y0, double y1, double dy,
		boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip);

	 boost::function<void()> netcdf_define(NcFile &nc, std::string const &generic_name) const;

};

}
