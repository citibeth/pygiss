# pyGISS: GISS Python Library
# Copyright (c) 2013 by Robert Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import string
import glob

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
def sum_by_cols(matrix) :
	return np.array(matrix.sum(axis=0)).reshape(-1)

def sum_by_rows(matrix) :
	return np.array(matrix.sum(axis=1)).reshape(-1)

# -----------------------------------------------------------
def multiglob_iterator(paths) :
	"""Iterator liss a bunch of files from a bunch of arguments.  Tries to work like ls
	Yields:
		(directory, filename) pairs
	See:
		lsacc.py
	"""
	if len(paths) == 0 :
		for fname in os.listdir('.') :
			yield ('', fname)
		return

	for path in paths :
		if os.path.isdir(path) :
			for fname in os.listdir(path) :
				yield (path, fname)

		elif os.path.isfile(path) :
			yield os.path.split(path)

		else :
			for ret in multiglob_iterator(glob.glob(path)) :
				yield ret
