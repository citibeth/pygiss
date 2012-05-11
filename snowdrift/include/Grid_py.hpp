#pragma once

#include <Python.h>

namespace giss {
	class Grid;
}

/// Classmembers of the Python class
struct GridDict {
	PyObject_HEAD
	giss::Grid *grid;	// The grid, in C++
};
