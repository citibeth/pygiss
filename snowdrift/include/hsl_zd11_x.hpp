namespace giss {

class ZD11_f;		// Fortran type, opaque
class ZD11;

}

extern "C" {
	void ZD11_c_init_(ZD11 &, ZD11_f &, int m, int n, int ne);
	int ZD11_put_type_c_(ZD11_f &, char *, int);
}

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

	// Set the type parameter in the ZD11 data structure
	int put_type(std::string const &str)
		{ return ZD11_put_type_c_(main, str.c_str(), str.length()); }
};

};
