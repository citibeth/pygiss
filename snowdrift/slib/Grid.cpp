#include <algorithm>
#include <boost/bind.hpp>
#include "Grid.hpp"
#include "ncutil.hpp"
#include <limits>
#include <CGAL/enum.h>
#include <cstdio>

#include "Grid_XY.hpp"
#include "Grid_LatLon.hpp"

namespace giss {


static double const nan = std::numeric_limits<double>::quiet_NaN();

Grid::Grid(std::string const &_stype, std::string const &_name, int _index_base, int _max_index) :
	stype(_stype), name(_name), _bounding_box_valid(false), index_base(_index_base), _rtree_valid(false), max_index(_max_index), index_size(max_index - index_base + 1)
{
fprintf(stderr, "Grid(this=%p)\n", this);
}


Grid::~Grid()
{
fprintf(stderr, "~Grid(this=%p)\n", this);
}


bool Grid::add_cell(GridCell const &gc)
{
	// Barf on degenerate grid cells.  They should already have
	// been clipped out.
	if (gc.poly.area() < 0) {
		std::cerr << "Polygon has area < 0: " << gc.poly << std::endl;
		throw std::exception();
	}

	_bounding_box_valid = false;
	_rtree_valid = false;
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

	NcVar *indexVar = nc->get_var((generic_name + ".realized_cells").c_str());
	NcVar *native_areaVar = nc->get_var((generic_name + ".native_area").c_str());
	NcVar *proj_areaVar = nc->get_var((generic_name + ".proj_area").c_str());



	// Write out the polygons
	NcVar *pointsVar = nc->get_var((generic_name + ".points").c_str());
	NcVar *polyVar = nc->get_var((generic_name + ".polygons").c_str());
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
		infoVar->add_att("max_index", max_index);

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
	auto polygonsVar = nc.add_var((generic_name + ".polygons").c_str(), ncInt, ncells_plus_1_Dim);
		polygonsVar->add_att("index_base", 0);

	return boost::bind(&Grid::netcdf_write, this, &nc, generic_name);
}

/** Lazily creates an RTree for the grid */
Grid::RTree &Grid::rtree()
{
//fprintf(stderr, "rtree() called\n");
	if (_rtree_valid) return *_rtree;
//fprintf(stderr, "rtree() recompute...\n");

	_rtree.reset(new Grid::RTree());
	// _rtree.RemoveAll();

	double min[2];
	double max[2];
	for (auto ii1=this->cells().begin(); ii1 != this->cells().end(); ++ii1) {
		GridCell const &gc = ii1->second;

		min[0] = CGAL::to_double(gc.bounding_box.xmin());
		min[1] = CGAL::to_double(gc.bounding_box.ymin());
		max[0] = CGAL::to_double(gc.bounding_box.xmax());
		max[1] = CGAL::to_double(gc.bounding_box.ymax());

//fprintf(stderr, "Adding bounding box: (%f %f)  (%f %f)\n", min[0], min[1], max[0], max[1]);

		// Deal with floating point...
		const double eps = 1e-7;
		double epsilon_x = eps * std::abs(max[0] - min[0]);
		double epsilon_y = eps * std::abs(max[1] - min[1]);
		min[0] -= epsilon_x;
		min[1] -= epsilon_y;
		max[0] += epsilon_x;
		max[1] += epsilon_y;

//std::cout << gc.poly << std::endl;
//std::cout << gc.bounding_box << std::endl;
//printf("(%g,%g) -> (%g,%g)\n", min[0], min[1], max[0], max[1]);
		_rtree->Insert(min, max, &gc);
	}


	_rtree_valid = true;
	return *_rtree;
}

// ----------------------------------------------------------------
static bool rasterize_point(
GridCell const *gc,			// Candidate grid cell to contain our x/y location
gc::Point_2 const *point,	// x/y location we're rasterizing
double **out_loc,			// Where to place the rasterized value
double const *data, int data_stride, int index_base,	// Where we read our values from
GridCell const **last_match)	// OUTPUT: Store here if we matched
{
	switch(gc->poly.oriented_side(*point)) {
		case CGAL::ON_POSITIVE_SIDE :
		case CGAL::ON_ORIENTED_BOUNDARY :
			**out_loc = data[(gc->index - index_base) * data_stride];
			*last_match = gc;
			return false;		// Stop searching
	}
	return true;		// Continue searching
}

/** @param nx Number of delta-x spaces between x0 and x1 */
void Grid::rasterize(
	double x0, double x1, int nx,
	double y0, double y1, int ny,
	double const *data, int data_stride,
	double *out, int xstride, int ystride)
{

	

//	// Convert dx,dy into nx,ny, i.e. the number of grid cells
//	int const nx = (int)((x1 - x0) / dx);
//	int const ny = (int)((y1 - y0) / dy);


	// Set up callback to call repeatedly
	char point_mem[sizeof(gc::Point_2)];
	double *out_loc;
	GridCell const *last_match = &_cells.begin()->second;
	auto callback = boost::bind(&rasterize_point, _1,
		(gc::Point_2 *)point_mem, &out_loc,
		data, data_stride, index_base,
		&last_match);

	Grid::RTree &rtree = this->rtree();

	double min[2];
	double max[2];
	for (int iy=0; iy < ny; ++iy) {
		const int indexy = iy * ystride;
		double y = y0 + (y1-y0) * ((double)iy / (double)(ny-1));
		min[1] = y; max[1] = y;
		for (int ix=0; ix < nx; ++ix) {
			const int index = indexy + (ix * xstride);
			double x = x0 + (x1-x0) * ((double)ix / (double)(nx-1));
			min[0] = x; max[0] = x;

			out_loc = &out[index];
			*out_loc = nan;

			// Figure out which grid cell we're in
			// Try the last grid cell we matched to
			gc::Point_2 *point = new (point_mem) gc::Point_2(x,y);
			switch(last_match->poly.oriented_side(*point)) {
				case CGAL::ON_POSITIVE_SIDE :
				case CGAL::ON_ORIENTED_BOUNDARY :
					*out_loc = data[(last_match->index - index_base) * data_stride];
				break;
				default :
					// No easy match, gotta go search the tree again.
					rtree.Search(min, max, callback);
				break;
			}
			point->~Point_2();
		}
	}
}
/** For use with Fortran */
extern "C"
void Grid_rasterize(
	Grid *grid,
	double x0, double x1, int nx,
	double y0, double y1, int ny,
	double const *data, int data_stride,
	double *out, int xstride, int ystride)
{
	grid->rasterize(x0, x1, nx, y0, y1, ny, data, data_stride, out, xstride, ystride);
}

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------






std::unique_ptr<Grid::SmoothingFunction> Grid::get_smoothing_function(std::set<int> const &mask)
{
	fprintf(stderr, "Grid::get_smoothing_function() undefined!\n");
	throw std::exception();
	//return std::unique_ptr<SmoothingFunction>();
}


std::unique_ptr<Grid> Grid::new_grid(std::string const &stype,
	std::string const &name, int index_base, int max_index)
{
	std::unique_ptr<Grid> grid;
	if (stype == "xy") {
		grid.reset(new Grid_XY(name, index_base, max_index));
	} else if (stype == "latlon") {
		grid.reset(new Grid_LatLon(name, index_base, max_index));
	} else {
		grid.reset(new Grid("generic", name, index_base, max_index));
	}
	return grid;
}


/** @param fname Name of file to load from (eg, an overlap matrix file)
@param vname Eg: "grid1" or "grid2" */
std::unique_ptr<Grid> Grid::netcdf_read(
//std::string const &fname,
NcFile &nc,
std::string const &vname)
{
	auto infoVar = nc.get_var((vname + ".info").c_str());
		int index_base = infoVar->get_att("index_base")->as_int(0);
		int max_index = infoVar->get_att("max_index")->as_int(0);
		std::string stype(infoVar->get_att("type")->as_string(0));

	std::unique_ptr<Grid> grid(new_grid(stype, vname, index_base, max_index));

	grid->read_from_netcdf(nc, vname);

	return grid;
}

/** @param fname Name of file to load from (eg, an overlap matrix file)
@param vname Eg: "grid1" or "grid2" */
void Grid::read_from_netcdf(
NcFile &nc,
std::string const &vname)
{
	// Read points 2-d array as single vector
	NcVar *vpoints = nc.get_var((vname + ".points").c_str());
	long npoints = vpoints->get_dim(0)->size();
	std::vector<double> points(npoints*2);
	vpoints->get(&points[0], npoints, 2);

	// Read the other simple vectors
	std::vector<int> polygons(read_int_vector(nc, vname + ".polygons"));
	int npoly = (int)polygons.size() - 1;
	NcVar *vpolygons = nc.get_var((vname + ".polygons").c_str());
	int poly_index_base = vpolygons->get_att("index_base")->as_int(0);

	std::vector<double> native_area(read_double_vector(nc, vname + ".native_area"));
	std::vector<int> indices(read_int_vector(nc, vname + ".realized_cells"));

	// Convert it into grid
	gc::Polygon_2 poly;
	for (int i=0; i<npoly; ++i) {
		poly.clear();
		for (int j=polygons[i] - poly_index_base; j < polygons[i+1] - poly_index_base; ++j) {
			poly.push_back(gc::Point_2(points[j*2], points[j*2+1]));
		}
		add_cell(GridCell(poly, indices[i], native_area[i]));
	}
}





}
