#pragma once

#include <boost/function.hpp>
#include <netcdfcpp.h>
#include "GridCell.hpp"

class NcFile;

namespace giss {

class Grid {


	std::map<int, GridCell> _cells;

	// Bounding box for overall grid after all cells have been added
	gc::Polygon_2 _bounding_box;

	bool _bounding_box_valid;


	void netcdf_write(NcFile *nc, std::string const &generic_name);

public:
//	enum class Type { XY, LATLON };
	// ---------------------------------------------------

	/** Type string identifying this kind of grid */
	std::string const stype;

	virtual ~Grid() {}


	virtual boost::function<void()> netcdf_define(NcFile &nc, std::string const &generic_name, std::string const &specific_name);

	Grid(std::string _stype) : stype(_stype), _bounding_box_valid(false) {}

	size_t size() const { return _cells.size(); }

	gc::Polygon_2 const &bounding_box();

	bool add_cell(GridCell const &gc);

	std::map<int, GridCell> const &cells() const
		{ return _cells; }

	GridCell const &operator[](int index)
		{ return _cells[index]; }
};

}