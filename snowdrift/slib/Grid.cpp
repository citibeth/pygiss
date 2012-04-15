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

void Grid::netcdf_write(NcFile *nc, std::string const &generic_name) const
{
	Grid const *grid = this;
	double const nan = std::numeric_limits<double>::quiet_NaN();
printf("AA1 %s\n", (name + ".realized_cells").c_str());

	NcVar *indexVar = nc->get_var((generic_name + ".realized_cells").c_str());
printf("AA2\n");
	NcVar *native_areaVar = nc->get_var((generic_name + ".native_area").c_str());
printf("AA2\n");
	NcVar *proj_areaVar = nc->get_var((generic_name + ".proj_area").c_str());

printf("AA2\n");


	// Write out the polygons
	NcVar *pointsVar = nc->get_var((generic_name + ".points").c_str());
	NcVar *polyVar = nc->get_var((generic_name + ".polygons").c_str());
printf("AA3\n");
	int ivert = 0;
	int i=0;
	for (auto ii = grid->_cells.begin(); ii != _cells.end(); ++ii, ++i) {
		GridCell const &gc(ii->second);

		indexVar->set_cur(i);
		indexVar->put(&gc.index, 1);
		native_areaVar->set_cur(i);
		native_areaVar->put(&gc.native_area, 1);
		proj_areaVar->set_cur(i);
		proj_areaVar->put(&gc.proj_area, 1);

		double point[2];
		polyVar->set_cur(i);
		polyVar->put(&ivert, 1);
		for (auto vertex = gc.poly.vertices_begin(); vertex != gc.poly.vertices_end(); ++vertex) {
			point[0] = CGAL::to_double(vertex->x());
			point[1] = CGAL::to_double(vertex->y());
			pointsVar->set_cur(ivert, 0);
			pointsVar->put(point, 1, 2);
			++ivert;
		}
	}

	// Write out a sentinel for polygon index bounds
	polyVar->set_cur(i);
	polyVar->put(&ivert, 1);
}

boost::function<void ()> Grid::netcdf_define(NcFile &nc, std::string const &generic_name) const
{
	auto oneDim = get_or_add_dim(nc, "one", 1);
	NcVar *infoVar = nc.add_var((generic_name + ".info").c_str(), ncInt, oneDim);
		infoVar->add_att("name", name.c_str());
		infoVar->add_att("type", stype.c_str());
		infoVar->add_att("index_base", index_base);

	// Allocate for the polygons
	int nvert = 0;
	for (std::map<int, GridCell>::const_iterator ii = _cells.begin(); ii != _cells.end(); ++ii) {
		GridCell const &gc(ii->second);
		nvert += gc.poly.size();
	}
	NcDim *ncellsDim = nc.add_dim((generic_name + ".num_realized_cells").c_str(), size());
	NcDim *ncells_plus_1_Dim = nc.add_dim((generic_name + ".num_realized_cells_plus1").c_str(), size()+1);
	NcDim *nvertDim = nc.add_dim((generic_name + ".num_vertices").c_str(), nvert);
	NcDim *twoDim = get_or_add_dim(nc, "two", 2);


	auto indexVar = nc.add_var((generic_name + ".realized_cells").c_str(), ncDouble, ncellsDim);
	indexVar->add_att("description",
		"Index of each realized grid cell, used to index into arrays"
		" representing values on the grid.");

	nc.add_var((generic_name + ".native_area").c_str(), ncDouble, ncellsDim);
	nc.add_var((generic_name + ".proj_area").c_str(), ncDouble, ncellsDim);


	nc.add_var((generic_name + ".points").c_str(), ncDouble, nvertDim, twoDim);
	auto polygonsVar = nc.add_var((generic_name + ".polygons").c_str(), ncDouble, ncells_plus_1_Dim);
		polygonsVar->add_att("index_base", 0);

	return boost::bind(&Grid::netcdf_write, this, &nc, generic_name);
}




}
