#include <Python.h>
#include <arrayobject.h>
#include <vector>
#include <string>

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
printf("BB1\n");
	if (!vec) {
		PyErr_SetString(PyExc_ValueError,
			"check_dimensions: Array object is null");
		return false;
	}
printf("BB1\n");

	int const ndim = dims.size();
	if (vec->descr->type_num != type_num || vec->nd != ndim)  {
		char buf[200];
		sprintf(buf, "check_dimensions: array must be of type_num %d and %d dimensions.", type_num, ndim);
		PyErr_SetString(PyExc_ValueError, buf);
		return false;
	}
printf("BB1\n");

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
printf("BB1\n");
	return true;
}


}
