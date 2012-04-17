#include <Python.h>
#include <arrayobject.h>

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

}
