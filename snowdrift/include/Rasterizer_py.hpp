#pragma once

#include <Python.h>

namespace giss {
	class Rasterizer;
}

/// Classmembers of the Python class
struct RasterizerDict {
	PyObject_HEAD
	giss::Rasterizer *main;	// The grid, in C++
};
