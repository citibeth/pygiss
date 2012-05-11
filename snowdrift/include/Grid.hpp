#pragma once

#include <boost/function.hpp>
#include <netcdfcpp.h>
#include "GridCell.hpp"
#include "RTree.h"
#include "SparseMatrix.hpp"

class NcFile;

namespace giss {


class Grid {
public:
	typedef ::RTree<GridCell const *, double, 2, double> RTree;

private:
	std::map<int, GridCell> _cells;

	// Bounding box for overall grid after all cells have been added
	gc::Polygon_2 _bounding_box;
	bool _bounding_box_valid;

	void netcdf_write(NcFile *nc, std::string const &generic_name) const;

	/** Used to do geometry calculations quickly on polygons in this grid */
	std::unique_ptr<Grid::RTree> _rtree;
	bool _rtree_valid;

public:
//	enum class Type { XY, LATLON };
	// ---------------------------------------------------

	std::map<int, GridCell>::const_iterator begin() { return _cells.begin(); }

	// Array base used for index numbers in _cells
	int const index_base;

	// Maximum value that index can have for any cell in the global grid, even if not realized in this Grid
	int const max_index;

	// Total number of grid cells in the indexing scheme
	int const index_size;

	std::string const name;	/// Name of this grid instance (eg: "ice", "gcm", etc)

	/** Type string identifying this kind of grid */
	std::string const stype;

	virtual ~Grid();

	virtual boost::function<void()> netcdf_define(NcFile &nc, std::string const &generic_name) const;

	Grid(std::string const &_stype, std::string const &_name, int _index_base, int _max_index);

	/** Lazily creates an RTree for the grid */
	Grid::RTree &rtree();
	

	size_t size() const { return _cells.size(); }

	gc::Polygon_2 const &bounding_box();

	bool add_cell(GridCell const &gc);

	std::map<int, GridCell> const &cells() const
		{ return _cells; }

	/** Correct by index_base here so indexing matches up with SparseMatrix.hpp */
	GridCell const &operator[](int index)
		{ return _cells[index + index_base]; }

	/** Factory method */
	static std::unique_ptr<Grid> new_grid(std::string const &type,
		std::string const &name, int index_base, int max_index);


	/** Main factory method */
	static std::unique_ptr<Grid> netcdf_read(NcFile &nc, std::string const &grid_var_name);

protected:
	virtual void read_from_netcdf(NcFile &nc, std::string const &grid_var_name);

public:

	/** Prepare for display of a function on this grid. */
	void rasterize(
		double x0, double x1, int nx,
		double y0, double y1, int ny,
		double const *data, int data_stride,
		double *out, int xstride, int ystride);


	struct SmoothingFunction {
		/** The Hessian matrix, used in QP */
		MapSparseMatrix H;

		/** Linear term to QP objective function.  Must be pairwise-multiplied
		by Z0, the HNTR regridding result.  Constant term is also available
		from this */
		std::vector<double> G0;

		SmoothingFunction(MapSparseMatrix &&_H) : H(_H), G0(H.ncol) {}
	};

	/** @param mask[size()] >=0 if we want to include this grid cell */
	virtual std::unique_ptr<Grid::SmoothingFunction> get_smoothing_function(std::set<int> const &mask);

#if 0
	/** Used to divide up a grid cell in the face of elevation classes */
	GridCell sub_by_elevation_class(
		GridCell const &gc0,
		int elevation_class,
		int min_elevation_class, int max_elevation_class)
	{
		int num_elevation_class = max_elevation_class - min_elevation_class + 1;
		int index1 = (gc0.index - index_base) * num_elevation_class
			+ (elevation_class - min_elevation_class) + index_base;
	}
#endif


};

}
