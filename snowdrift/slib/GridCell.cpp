#include "GridCell.hpp"

namespace giss {

GridCell::GridCell(gc::Polygon_2 const &_poly, double _index, double _native_area) :
	poly(_poly),
	bounding_box(CGAL::bounding_box(poly.vertices_begin(), poly.vertices_end())),
	index(_index), native_area(_native_area),
	proj_area(CGAL::to_double(poly.area()))
{}


}
