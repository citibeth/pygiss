#pragma once

#include <string>

namespace giss {

class ZD11_f;		// Fortran type, opaque
class ZD11;

}

extern "C" int ZD11_c_init_(giss::ZD11 self, giss::ZD11_f &main, int m, int n, int ne);
extern "C" int ZD11_put_type_c_(giss::ZD11_f &, char const *, int);

namespace giss {

// =============== C++ Peer Classes
class ZD11 {
public :
	ZD11_f &main;		// Actual storage for this

	int &m;
	int &n;
	int &ne;				// Number of non-zero elements

	int * const row;		// int[m]
	int * const col;		// int[n]
	double * const val;		// double[ne]

	ZD11() : main(*(ZD11_f *)0), m(*(int *)0), n(*(int *)0), ne(*(int *)0),
		row(0), col(0), val(0) {}

	// Set the type parameter in the ZD11 data structure
	int put_type(std::string const &str)
		{ return ZD11_put_type_c_(main, str.c_str(), str.length()); }
};

};
