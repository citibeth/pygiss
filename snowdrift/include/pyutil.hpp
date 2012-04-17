#pragma once

namespace giss {

bool is_doublevector(PyArrayObject *vec);

bool check_dimensions(PyArrayObject *vec, int ndim, int *dims);

}
