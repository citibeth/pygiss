import numpy as np
import os
import string

class Struct:
	"""Convert a dict() to a struct."""
	def __init__(self, entries): 
		self.__dict__.update(entries)

class curry:
	"""Curry a function, i.e. produce a new function in which the
	first n parameters have been set.

	Args:
		fun:
			The function to curry
		*args:
			The first few arguments to the function
		**kwargs:
			Any keyword arguments to pass to the function

	Returns:
		Object that acts like the new curried function.

	Example:
		double = curry(operator.mul, 2)
		print double(17)
		triple = curry(operator.mul, 3)

	See:
		http://code.activestate.com/recipes/52549-curry-associating-parameters-with-a-function"""

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

# -----------------------------------------------------------
def numpy_stype(var) :
	"""Provides a summary of a numpy variable.

	Returns:
		Textual summary.  Eg: int[4,3]
	"""
	return '%s%s' % (str(var.dtype), str(var.shape))
# -----------------------------------------------------------
def search_file(filename, search_path):
	"""Given a search path, find file.
	Args:
		Alt 1: search_path[] (string):
			Directories where to search for file
		Alt 2: search_path (string)
			Directories where to search for file, using path separator.
			Eg: '/usr/home:/usr/bin'

	See:
		http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
	"""
	if isinstance(search_path, str) :
		search_path = string.split(search_path, os.pathsep)
	for path in search_path :
		if os.path.exists(os.path.join(path, filename)):
			return os.path.abspath(os.path.join(path, filename))

	# Not found :(
	return None
# -----------------------------------------------------------
