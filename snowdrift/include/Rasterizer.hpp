#pragma once

#include <vector>
#include <blitz/array.h>

namespace giss {

class HeightClassifier;
class Grid;

/** Used to rasterize a set of values on a grid, for display */
class Rasterizer {

	friend void rasterize(
	Rasterizer const &rast1,
	blitz::Array<double,1> const &Z1,
	blitz::Array<double,2> &Z1_r);

	friend void rasterize_mask(
	Rasterizer const &rast1,
	Rasterizer const &rast2,
	blitz::Array<double,1> const &Z1,
	blitz::Array<int,1> const &mask2,
	blitz::Array<double,2> &Z1_r);

	friend void rasterize_hc(
	Rasterizer const &rast1,
	Rasterizer const &rast2,
	std::vector<blitz::Array<double,1>> const &Z1,
	blitz::Array<double,1> const &elevation2,
	blitz::Array<int,1> const &mask2,
	HeightClassifier &height_classifier,
	blitz::Array<double,2> &Z1_r);

public:
	// Description of grid to rasterize from
	const int grid_index_size;		// from grid->index_size

	// Description of grid to rasterize to
	const double x0, x1;
	const int nx;
	const double y0, y1;
	const int ny;

private:
	/// The point at (x,y) needs to access grid indexed by this (normalized to index_base=0)
	blitz::Array<int,2> index;
public :

	Rasterizer(
		Grid *grid,
		double _x0, double _x1, int _nx,	// Grid to rasterize to
		double _y0, double _y1, int _ny);

};

void rasterize(
Rasterizer const &rast1,
blitz::Array<double,1> const &Z1,
blitz::Array<double,2> &Z1_r);

void rasterize_mask(
Rasterizer const &rast1,
Rasterizer const &rast2,
blitz::Array<double,1> const &Z1,
blitz::Array<int,1> const &mask2,
blitz::Array<double,2> &Z1_r);

/** @param rast1 GCM (grid1) rasterizer --- the main grid we're rasterizing
@param rast2 ice (grid2) rasterizer: provides elevation class
@param height_max1 Elevation class definitions per GCM grid cell */
void rasterize_hc(
Rasterizer const &rast1,
Rasterizer const &rast2,
std::vector<blitz::Array<double,1>> const &Z1,
blitz::Array<double,1> const &elevation2,
blitz::Array<int,1> const &mask2,
HeightClassifier &height_classifier,
blitz::Array<double,2> &Z1_r);


}
