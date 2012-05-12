#include <limits>
#include "Rasterizer.hpp"
#include "Grid.hpp"
#include "HeightClassifier.hpp"

namespace giss {

static double const nan = std::numeric_limits<double>::quiet_NaN();

/** Function used for callback in RTree search */
static bool rasterize_point(
GridCell const *gc,			// Candidate grid cell to contain our x/y location
gc::Point_2 const *point,	// x/y location we're rasterizing
int **index_loc,			// Where to place the rasterized value
int grid_index_base,				// grid->index_base
GridCell const **last_match)	// OUTPUT: Store here if we matched
{
	switch(gc->poly.oriented_side(*point)) {
		case CGAL::ON_POSITIVE_SIDE :
		case CGAL::ON_ORIENTED_BOUNDARY :
			**index_loc = (gc->index - grid_index_base);
			*last_match = gc;
			return false;		// Stop searching
	}
	return true;		// Continue searching
}


Rasterizer::Rasterizer(
	Grid *grid,
	double _x0, double _x1, int _nx,	// Grid to rasterize to
	double _y0, double _y1, int _ny) :
grid_index_size(grid->index_size),
x0(_x0), x1(_x1), nx(_nx),
y0(_y0), y1(_y1), ny(_ny),
index(nx, ny)
{
	// Set up callback to call repeatedly
	char point_mem[sizeof(gc::Point_2)];	// Create a gc::Point_2 in here
	int *index_loc;
	GridCell const *last_match = &grid->begin()->second;
	auto callback = boost::bind(&rasterize_point, _1,
		(gc::Point_2 *)point_mem, &index_loc,
		grid->index_base, &last_match);

	Grid::RTree &rtree = grid->rtree();

	double min[2];
	double max[2];
	for (int iy=0; iy < ny; ++iy) {
		double y = y0 + (y1-y0) * ((double)iy / (double)(ny-1));
		min[1] = y; max[1] = y;
		for (int ix=0; ix < nx; ++ix) {
			double x = x0 + (x1-x0) * ((double)ix / (double)(nx-1));
			min[0] = x; max[0] = x;

			index_loc = &index(ix,iy);
			*index_loc = -1;

			// Figure index which grid cell we're in
			// Try the last grid cell we matched to
			gc::Point_2 *point = new (point_mem) gc::Point_2(x,y);
			switch(last_match->poly.oriented_side(*point)) {
				case CGAL::ON_POSITIVE_SIDE :
				case CGAL::ON_ORIENTED_BOUNDARY :
					*index_loc = (last_match->index - grid->index_base);
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

void rasterize(
Rasterizer const &rast1,
blitz::Array<double,1> const &Z1,
blitz::Array<double,2> &Z1_r)
{
	// --------- Check array dimensions
	if (Z1_r.extent(0) != rast1.nx || Z1_r.extent(1) != rast1.ny) {
		fprintf(stderr, "Bad array dimension Z1_r(%d, %d) should be (%d, %d)\n", Z1_r.extent(0), Z1_r.extent(1), rast1.nx, rast1.ny);
		throw std::exception();
	}
	if (Z1.extent(0) != rast1.grid_index_size) {
		fprintf(stderr, "Bad array dimension Z1(%d) should be (%d)\n", Z1.extent(0), rast1.grid_index_size);
		throw std::exception();
	}

	// ----------- Rasterize!
	for (int ix=0; ix<rast1.nx; ++ix) {
		for (int iy=0; iy<rast1.ny; ++iy) {
			int idx = rast1.index(ix,iy);
			Z1_r(ix,iy) = (idx < 0 ? nan : Z1(rast1.index(ix,iy)));
		}
	}
}

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
blitz::Array<double,2> &Z1_r)
{
	// --------- Check array dimensions
	if (Z1_r.extent(0) != rast1.nx || Z1_r.extent(1) != rast1.ny) {
		fprintf(stderr, "Bad array dimension Z1_r(%d, %d) should be (%d, %d)\n", Z1_r.extent(0), Z1_r.extent(1), rast1.nx, rast1.ny);
		throw std::exception();
	}
	if (Z1[0].extent(0) != rast1.grid_index_size) {
		fprintf(stderr, "Bad array dimension Z1(%d) should be (%d)\n", Z1[0].extent(0), rast1.grid_index_size);
		throw std::exception();
	}
	if (elevation2.extent(0) != rast2.grid_index_size) {
		fprintf(stderr, "Bad array dimension elevation2(%d) should be (%d)\n", elevation2.extent(0), rast2.grid_index_size);
	}
	if (height_classifier.size() != rast1.grid_index_size) {
		fprintf(stderr, "Bad array dimension height_classifier(%d) should be (%d)\n", height_classifier.size(), rast1.grid_index_size);
	}
	if (rast1.nx != rast2.nx || rast1.ny != rast2.ny) {
		fprintf(stderr, "rast1(%d,%d) and rast2(%d,%d) must have same dimensions\n",
			rast1.nx, rast1.ny, rast2.nx, rast2.ny);
	}

	// ----------- Rasterize!
	for (int ix=0; ix<rast1.nx; ++ix) {
		for (int iy=0; iy<rast1.ny; ++iy) {
			// Figure out which gridcell this pixel overlaps in grid1 and grid2
			int i1 = rast1.index(ix, iy);
			int i2 = rast2.index(ix, iy);

			if (mask2(i2) == 0) {
				Z1_r(ix,iy) = nan;
			} else {
				// Figure out its height class
				int hclass = height_classifier.get_hclass(i1, elevation2(i2));

				// Look up and store its value
				Z1_r(ix,iy) = Z1[hclass](i1);
			}
		}
	}
}

}
