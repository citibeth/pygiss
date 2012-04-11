#pragma once

#include <netcdfcpp.h>

namespace giss {

inline NcDim *get_or_add_dim(NcFile &nc, char *dim_name, long dim_size)
{
	// Look up dim the slow way...
	int num_dims = nc.num_dims();
	for (int i=0; i<num_dims; ++i) {
		NcDim *dim = nc.get_dim(i);
		char const *name = dim->name();

		if (strcmp(name, dim_name) == 0) {
			long sz = dim->size();
			if (sz != dim_size) {
				fprintf(stderr, "Error: dimension %s (size = %ld) being redefined to size = %ld\n",
					dim_name, sz, dim_size);
				throw std::exception();
			}
			return dim;
		}
	}

	return nc.add_dim(dim_name, dim_size);
}

}
