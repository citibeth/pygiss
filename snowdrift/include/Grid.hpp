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


	void netcdf_write(NcFile *nc, std::string const &generic_name) const;

public:
//	enum class Type { XY, LATLON };
	// ---------------------------------------------------

	std::string const name;	/// Name of this grid instance (eg: "ice", "gcm", etc)

	/** Type string identifying this kind of grid */
	std::string const stype;

	virtual ~Grid() {}


	virtual boost::function<void()> netcdf_define(NcFile &nc, std::string const &generic_name) const;

	Grid(std::string const &_stype, std::string const &_name) :
		stype(_stype), name(_name), _bounding_box_valid(false) {}

	size_t size() const { return _cells.size(); }

	gc::Polygon_2 const &bounding_box();

	bool add_cell(GridCell const &gc);

	std::map<int, GridCell> const &cells() const
		{ return _cells; }

	GridCell const &operator[](int index)
		{ return _cells[index]; }
};

}
