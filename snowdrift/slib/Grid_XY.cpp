#include <boost/bind.hpp>
#include "Grid_XY.hpp"

namespace giss {

/**
@var xb x-axis boundaries of grid cells --- last one is 360 + first.
@var yb y-axis boundaries of grid cells --- sorted from low to high
@var proj Projection we will use to put this on a 2-D map
@var clip_poly Only realize grid cells that intersect with this polygon (on the map)
*/
std::unique_ptr<Grid_XY> Grid_XY::new_grid(
	std::string const &name,
	std::vector<double> const &xb,
	std::vector<double> const &yb,
	boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip)

{
	std::unique_ptr<Grid_XY> grid(new Grid_XY(name));
	grid->x_boundaries = xb;
	grid->y_boundaries = yb;

	// Set up the main grid
	int index = 1;
	for (int iy = 0; iy < yb.size()-1; ++iy) {
		double y0 = yb[iy];
		double y1 = yb[iy+1];
//		grid->y_centers.push_back(.5*(y0+y1));

		for (int ix = 0; ix < xb.size()-1; ++ix, ++index) {
			double x0 = xb[ix];
			double x1 = xb[ix+1];
//			grid->x_centers.push_back(.5*(x0+x1));

			gc::Polygon_2 poly;
			poly.push_back(gc::Point_2(x0,y0));
			poly.push_back(gc::Point_2(x1,y0));
			poly.push_back(gc::Point_2(x1,y1));
			poly.push_back(gc::Point_2(x0,y1));

			// Don't include things outside our clipping region
			if (!euclidian_clip(poly)) continue;

//std::cout << "ix=" << ix << " x0=" << x0 << ", iy=" << iy << ", poly=" << poly << std::endl;

			grid->add_cell(GridCell(poly, index, (x1-x0)*(y1-y0)));
		}
	}

#if 0    // Not needed, this is done automatically
	// Set up the bounding box
	// Overall bounding box of ice grid
	grid->bounding_box.reset(new gc::Polygon_2);
	grid->bounding_box->push_back(gc::Point_2(xb[0], yb[0]));
	grid->bounding_box->push_back(gc::Point_2(xb.back(), yb[0]));
	grid->bounding_box->push_back(gc::Point_2(xb.back(), yb.back()));
	grid->bounding_box->push_back(gc::Point_2(xb[0], yb.back()));
#endif

	return grid;
}

/** @var nx Number of grid cells in x direction */
std::unique_ptr<Grid_XY> Grid_XY::new_grid(
	std::string const &name,
	double x0, double x1, double dx,
	double y0, double y1, double dy,
	boost::function<bool(gc::Polygon_2 const &)> const &euclidian_clip)
{
	// Convert dx,dy into nx,ny, i.e. the number of grid cells
	int const nx = (int)((x1 - x0) / dx);
	int const ny = (int)((y1 - y0) / dy);


	// Set up x
	std::vector<double> xb;
	double nx_inv = 1.0 / (double)nx;
	for (int i=0; i<nx; ++i)
		xb.push_back(x0 + (x1 - x0) * ((double)i) * nx_inv);
	xb.push_back(x1);

	// Set up y
	std::vector<double> yb;
	double ny_inv = 1.0 / (double)ny;
	for (int i=0; i<ny; ++i)
		yb.push_back(y0 + (y1 - y0) * (double)i * ny_inv);
	yb.push_back(y1);

	return new_grid(name, xb, yb, euclidian_clip);
}

static void Grid_XY_netcdf_write(
	boost::function<void()> const &parent,
	NcFile *nc, Grid_XY const *grid, std::string const &generic_name)
{
	parent();

	NcVar *xbVar = nc->get_var((generic_name + ".x_boundaries").c_str());
	NcVar *ybVar = nc->get_var((generic_name + ".y_boundaries").c_str());

	xbVar->put(&grid->x_boundaries[0], grid->x_boundaries.size());
	ybVar->put(&grid->y_boundaries[0], grid->y_boundaries.size());
}

boost::function<void ()> Grid_XY::netcdf_define(NcFile &nc, std::string const &generic_name) const
{
	auto parent = Grid::netcdf_define(nc, generic_name);

	NcDim *xbDim = nc.add_dim((generic_name + ".x_boundaries.length").c_str(),
		this->x_boundaries.size());
	NcVar *xbVar = nc.add_var((generic_name + ".x_boundaries").c_str(),
		ncDouble, xbDim);
	NcDim *ybDim = nc.add_dim((generic_name + ".y_boundaries.length").c_str(),
		this->y_boundaries.size());
	NcVar *ybVar = nc.add_var((generic_name + ".y_boundaries").c_str(),
		ncDouble, ybDim);

	return boost::bind(&Grid_XY_netcdf_write, parent, &nc, this, generic_name);
}



}
