#pragma once

#include "geometry.hpp"

namespace giss {

struct GridCell {
	gc::Polygon_2 const poly;			/// The polygon representing the grid cell (on the map)
	gc::Iso_rectangle_2 const bounding_box;	/// Bounding box of the polygon

	/** A canonical index used to uniquely identify which gridcell this is in the grid. */
	int const index;

	/** Exact area of the grid cell in its native geometry (eg, area
	of a lat/lon gridcell on the sphere) */
	double const native_area;

	/// Actual area of the grid cell, once projected and approximated with poloygons
	double const proj_area;

	/** Makes an invalid "dummy" grid cell */
	GridCell() : index(-1), native_area(0), proj_area(0) {}

	/**
	@param _poly Polygon describing this grid cell geometrically.
	@param _native_area Area of this grid cell in its native (un-projected) geometry.
	@param _index Canonical identifier for this grid cell in its grid. */
	GridCell(gc::Polygon_2 const &_poly, double _index, double _native_area);
};

}
