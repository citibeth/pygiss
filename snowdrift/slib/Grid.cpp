#include <algorithm>
#include <boost/bind.hpp>
#include "Grid.hpp"
#include "ncutil.hpp"
#include <limits>

namespace giss {

bool Grid::add_cell(GridCell const &gc)
{
	// Barf on degenerate grid cells.  They should already have
	// been clipped out.
	if (gc.poly.area() < 0) {
		std::cerr << "Polygon has area < 0: " << gc.poly << std::endl;
		throw std::exception();
	}

	_bounding_box_valid = false;
	_cells.insert(std::make_pair(gc.index, gc));
	return true;
}


gc::Polygon_2 const &Grid::bounding_box() {		// lazy eval
	if (_bounding_box_valid) return _bounding_box;

	// Recompute bounding box and return
	// Be lazy, base bounding box on minimum and maximum values in points
	// (instead of computing the convex hull)
	gc::Kernel::FT minx(1e100);
	gc::Kernel::FT maxx(-1e100);
	gc::Kernel::FT miny(1e100);
	gc::Kernel::FT maxy(-1e100);
	_bounding_box.clear();
	for (auto ii = _cells.begin(); ii != _cells.end(); ++ii) {
		GridCell const &gc(ii->second);

		for (auto vertex = gc.poly.vertices_begin(); vertex != gc.poly.vertices_end(); ++vertex) {
			minx = std::min(minx, vertex->x());
			maxx = std::max(maxx, vertex->x());
			miny = std::min(miny, vertex->y());
			maxy = std::max(maxy, vertex->y());
		}
	}

	// Set up the bounding box as a simple polygon
	_bounding_box.push_back(gc::Point_2(minx, miny));
	_bounding_box.push_back(gc::Point_2(maxx, miny));
	_bounding_box.push_back(gc::Point_2(maxx, maxy));
	_bounding_box.push_back(gc::Point_2(minx, maxy));

	// Return our newly computed bounding box
	_bounding_box_valid = true;
	return _bounding_box;
}

void Grid::netcdf_write(NcFile *nc, std::string const &generic_name)
{
	Grid *grid = this;
	double const nan = std::numeric_limits<double>::quiet_NaN();

	// Write out the polygons
	NcVar *polyVar = nc->get_var((generic_name + ".polygons").c_str());
	int ivert = 0;
	for (auto ii = grid->_cells.begin(); ii != _cells.end(); ++ii) {
		GridCell const &gc(ii->second);

		double point[2];
		for (auto vertex = gc.poly.vertices_begin(); vertex != gc.poly.vertices_end(); ++vertex) {
			point[0] = CGAL::to_double(vertex->x());
			point[1] = CGAL::to_double(vertex->y());
			polyVar->set_cur(ivert, 0);
			polyVar->put(point, 1, 2);
			++ivert;
		}

		// First point again
		auto vertex = gc.poly.vertices_begin();
		point[0] = CGAL::to_double(vertex->x());
		point[1] = CGAL::to_double(vertex->y());
		polyVar->set_cur(ivert, 0);
		polyVar->put(point, 1, 2);
		++ivert;

		// Spacer between points
		point[0] = nan;
		point[1] = nan;
		polyVar->set_cur(ivert, 0);
		polyVar->put(point, 1, 2);
		++ivert;
	}
}

boost::function<void ()> Grid::netcdf_define(NcFile &nc, std::string const &generic_name, std::string const &specific_name)
{
	auto oneDim = get_or_add_dim(nc, "one", 1);
	NcVar *infoVar = nc.add_var((generic_name + ".info").c_str(), ncInt, oneDim);
		infoVar->add_att("name", specific_name.c_str());

	infoVar->add_att("type", stype.c_str());

	// Allocate for the polygons
	int nvert = 0;
	for (std::map<int, GridCell>::iterator ii = _cells.begin(); ii != _cells.end(); ++ii) {
		GridCell const &gc(ii->second);

		// One extra vertex to close the polygon.  One extra vertex for nan spacer
		nvert += gc.poly.size() + 2;
	}
	NcDim *nvertDim = nc.add_dim((generic_name + ".num_vertices").c_str(), nvert);
	NcDim *twoDim = get_or_add_dim(nc, "two", 2);
	nc.add_var((generic_name + ".polygons").c_str(), ncDouble, nvertDim, twoDim);

	return boost::bind(&Grid::netcdf_write, this, &nc, generic_name);
}




}
