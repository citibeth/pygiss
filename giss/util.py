import numpy as np

class Struct:
    def __init__(self, entries): 
        self.__dict__.update(entries)

# See: http://code.activestate.com/recipes/52549-curry-associating-parameters-with-a-function/
#double = curry(operator.mul, 2)
#triple = curry(operator.mul, 3)
class curry:
	def __init__(self, fun, *args, **kwargs):
		self.fun = fun
		self.pending = args[:]
		self.kwargs = kwargs.copy()
	def __call__(self, *args, **kwargs):
		if kwargs and self.kwargs:
			kw = self.kwargs.copy()
			kw.update(kwargs)
		else:
			kw = kwargs or self.kwargs
		return self.fun(*(self.pending + args), **kw)

# Prints a summary of a numpy array
def numpy_stype(var) :
	return ''.join([str(var.dtype), str(var.shape)])

# Efficiently reads an array from a netCDF file
# @param dtype Data type of output array
def read_ncvar(nc, varname, dtype=float) :
	ncvar = nc.variables[varname]
	ret = np.zeros(ncvar.shape,dtype=dtype)
	ret[:] = ncvar[:]
	return ret
