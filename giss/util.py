import numpy as np
import os
import string

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

# Find a file given a search path (Python recipe)
# http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
def search_file(filename, search_path):
	"""Given a search path, find file
	"""
	paths = string.split(search_path, os.pathsep)
	for path in paths:
		if os.path.exists(os.path.join(path, filename)):
			return os.path.abspath(os.path.join(path, filename))

	# Not found :(
	return None

# Finds a file relative to the DATA_PATH environment variable
# (home for all our auxillary data)
def find_data_file(filename) :
	# Don't mess with absolute pathnames
	if os.path.isabs(filename) : return filename

	# Try full name on the search path
	if 'DATA_PATH' in os.environ :
		search_path = os.environ['DATA_PATH']
	else :
		search_path = '.' + os.pathsep + os.path.join(os.environ['HOME'], 'data')
	ret = search_file(filename, search_path)
	if ret is not None : return ret

	# Try leafname on the search path
	leaf = os.path.basename(filename)
	ret = search_file(leaf, search_path)
	if ret is not None : return ret

	raise Exception('Cannot find file %s in search path %s' % (filename, search_path))
# -----------------------------------------------------------
# Check dimensions of a numpy variable
def check_shape(var, dims, varname) :
	if var.shape != dims :
		raise Exception('%s%s should have dimensions %s' % (varname, str(dims), str(var.shape)))

# Check length of a non-numpy variable
def check_len(var, llen, varname) :
	if len(var) != llen :
		raise Exception('%s(%d) should have length %d' % (varname, len(var), llen))
# -----------------------------------------------------------------
