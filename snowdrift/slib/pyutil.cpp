#include <Python.h>
#include <arrayobject.h>
#include <vector>
#include <string>
#include "pyutil.hpp"

namespace giss {

/** Check that PyArrayObject is a double (Float) type and a vector
@return 0 if not a double vector, and also raise exception. */
bool is_doublevector(PyArrayObject *vec)  {
	if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
		PyErr_SetString(PyExc_ValueError,
			"In is_doublevector: array must be of type Float and 1 dimensional (n).");
		return false;
	}
	return true;
}

/** @param type_num See http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html eg: NPY_DOUBLE */
bool check_dimensions(PyArrayObject *vec, std::string const &vec_name, int type_num, 
std::vector<int> const &dims)
{
	if (!vec) {
		PyErr_SetString(PyExc_ValueError,
			"check_dimensions: Array object is null");
		return false;
	}

	int const ndim = dims.size();
	if (vec->descr->type_num != type_num || vec->nd != ndim)  {
		char buf[200];
		sprintf(buf, "check_dimensions: %s must be of type_num %d and %d dimensions (its is of type_num=%d and %d dimensions).", vec_name.c_str(), type_num, ndim, vec->descr->type_num, vec->nd);
		PyErr_SetString(PyExc_ValueError, buf);
		return false;
	}

	for (int i=0; i<dims.size(); ++i) {
		if (dims[i] < 0) continue;		// Don't check this dimension
		if (dims[i] != vec->dimensions[i]) {
			char buf[200];
			sprintf(buf,
				"%s: Array dimension #%d is %d, should be %d",
				vec_name.c_str(), i, vec->dimensions[i], dims[i]);
			PyErr_SetString(PyExc_ValueError, buf);
			return false;
		}
	}
	return true;
}

/** Used to pull apart a Numpy array into an array of arrays, one per
height class.
@param Z1_py The numpy data.  dimension[0] = n, dimension[1] = num_hclass */
std::vector<blitz::Array<double,1>> py_to_blitz_Z1(PyArrayObject *Z1_py)
{
	int num_hclass = Z1_py->dimensions[1];

	// ======== Convert to Blitz++ arrays
    blitz::TinyVector<int,1> shape(0);
		shape[0] = Z1_py->dimensions[0];
    blitz::TinyVector<int,1> strides(0);
		strides[0] = Z1_py->strides[0] / sizeof(double);
	std::vector<blitz::Array<double,1>> Z1;
	for (int heighti = 0; heighti < num_hclass; ++heighti) {
		Z1.push_back(blitz::Array<double,1>(
			(double *)Z1_py->data + (Z1_py->strides[1] / sizeof(double)) * heighti,
			shape, strides, blitz::neverDeleteData));
	}
	return Z1;
}



}
