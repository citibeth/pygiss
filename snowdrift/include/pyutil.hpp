#pragma once

#include <vector>
#include <blitz/array.h>

namespace giss {

bool is_doublevector(PyArrayObject *vec);

bool check_dimensions(PyArrayObject *vec, std::string const &vec_name,
	int type_num, std::vector<int> const &dims);

/** Convert a Numpy array to a blitz one, using the original's data (no copy).
N is the rank of the array.
@see: http://mail.scipy.org/pipermail/numpy-discussion/2004-March/002892.html
*/
template<class T, int N>
blitz::Array<T,N> py_to_blitz(PyArrayObject* arr_obj)
{
    int T_size = sizeof(T);
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    npy_intp *arr_dimensions = arr_obj->dimensions;
    npy_intp *arr_strides = arr_obj->strides;

    for (int i=0;i<N;++i) {
        shape[i]   = arr_dimensions[i];
		// Python/Numpy strides are in bytes, Blitz++ in sizeof(T) units.
        strides[i] = arr_strides[i]/T_size;
    }
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
		blitz::neverDeleteData);
}


/** Used to pull apart a Numpy array into an array of arrays, one per
height class.
@param Z1_py The numpy data.  dimension[0] = n, dimension[1] = num_hclass */
std::vector<blitz::Array<double,1>> py_to_blitz_Z1(PyArrayObject *Z1_py);


}
