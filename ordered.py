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

from UserDict import DictMixin

# Python ordered dictionary
# See: 

class odict(DictMixin):

	"""Dictionary in which the insertion order of items is preserved
	(using an internal double linked list). In this implementation
	replacing an existing item keeps it at its original position.

	Internal representation: values of the dict:
		[pred_key, val, succ_key]

	The sequence of elements uses as a double linked list. The links
	are dict keys. self.lh and self.lt are the keys of first and last
	element inseted in the odict. In a C reimplementation of this data
	structure, things can be simplified (and speed up) a lot if given
	a value you can at the same time find its key. With that, you can
	use normal C pointers.

	Usage:
		Import and create ordered dictionary:

			>>> import ordered
			>>> od = ordered.odict()

		type conversion to ordinary dict. This will fail:

			>>> dict(odict([(1, 1)]))
			{1: [nil, 1, nil]}

		The reason for this is here -> http://bugs.python.org/issue1615701

		The __init__ function of dict checks wether arg is subclass of
		dict, and ignores overwritten __getitem__ & co if so.

		This was fixed and later reverted due to behavioural problems with pickle.

		Use one of the following ways for type conversion:

			>>> dict(odict([(1, 1)]).items())
			{1: 1}

			>>> odict([(1, 1)]).as_dict()
			{1: 1}

		It is possible to use abstract mixin class _odict to hook
		another dict base implementation. This is useful i.e. when
		persisting to ZODB. Inheriting from dict and Persistent at the
		same time fails:

		>>> from persistent.dict import PersistentDict
		>>> class podict(_odict, PersistentDict):
		...		def _dict_impl(self):
		...			return PersistentDict
		
	See:
		http://pypi.python.org/pypi/odict/
	"""
 
	def __init__(self, data=None, **kwdata):
		self._keys = []
		self._data = {}
		if data is not None:
			if hasattr(data, 'items'):
				items = data.items()
			else:
				items = list(data)
			for i in xrange(len(items)):
				length = len(items[i])
				if length != 2:
					raise ValueError('dictionary update sequence element '
						'#%d has length %d; 2 is required' % (i, length))
				self._keys.append(items[i][0])
				self._data[items[i][0]] = items[i][1]
		if kwdata:
			self._merge_keys(kwdata.iterkeys())
			self.update(kwdata)


	def __repr__(self):
		result = []
		for key in self._keys:
			result.append('(%s, %s)' % (repr(key), repr(self._data[key])))
		return ''.join(['OrderedDict', '([', ', '.join(result), '])'])


	def _merge_keys(self, keys):
		self._keys.extend(keys)
		newkeys = {}
		self._keys = [newkeys.setdefault(x, x) for x in self._keys
			if x not in newkeys]

	def __iter__(self):
		for key in self._keys:
			yield key


	def update(self, data):
		if data is not None:
			if hasattr(data, 'iterkeys'):
				self._merge_keys(data.iterkeys())
			else:
				self._merge_keys(data.keys())
			self._data.update(data)



	def __setitem__(self, key, value):
		if key not in self._data:
			self._keys.append(key)
		self._data[key] = value
		
		
	def __getitem__(self, key):
		if isinstance(key, slice):
			result = [(k, self._data[k]) for k in self._keys[key]]
			return OrderedDict(result)
		return self._data[key]
	
	
	def __delitem__(self, key):
		del self._data[key]
		self._keys.remove(key)
		
		
	def keys(self):
		return list(self._keys)
	
	
	def copy(self):
		copyDict = odict()
		copyDict._data = self._data.copy()
		copyDict._keys = self._keys[:]
		return copyDict


# =============================================================
import collections

KEY, PREV, NEXT = range(3)

class oset(collections.MutableSet):
	"""Set that remembers original insertion order.

	Runs on Py2.6 or later (and runs on 3.0 or later without any
	modifications).

	Implementation based on a doubly linked link and an internal
	dictionary. This design gives OrderedSet the same big-Oh running
	times as regular sets including O(1) adds, removes, and lookups as
	well as O(n) iteration.

	See:
		http://code.activestate.com/recipes/576694/

	doctest:
		>>> import ordered
		>>> print(ordered.oset('abracadaba'))
		oset(['a', 'b', 'r', 'c', 'd'])
	"""

	def __init__(self, iterable=None):
		self.end = end = [] 
		end += [None, end, end]			# sentinel node for doubly linked list
		self.map = {}					# key --> [key, prev, next]
		if iterable is not None:
			self |= iterable

	def __len__(self):
		return len(self.map)

	def __contains__(self, key):
		return key in self.map

	def add(self, key):
		if key not in self.map:
			end = self.end
			curr = end[PREV]
			curr[NEXT] = end[PREV] = self.map[key] = [key, curr, end]

	def discard(self, key):
		if key in self.map:		   
			key, prev, next = self.map.pop(key)
			prev[NEXT] = next
			next[PREV] = prev

	def __iter__(self):
		end = self.end
		curr = end[NEXT]
		while curr is not end:
			yield curr[KEY]
			curr = curr[NEXT]

	def __reversed__(self):
		end = self.end
		curr = end[PREV]
		while curr is not end:
			yield curr[KEY]
			curr = curr[PREV]

	def pop(self, last=True):
		if not self:
			raise KeyError('set is empty')
		key = next(reversed(self)) if last else next(iter(self))
		self.discard(key)
		return key

	def __repr__(self):
		if not self:
			return '%s()' % (self.__class__.__name__,)
		return '%s(%r)' % (self.__class__.__name__, list(self))

	def __eq__(self, other):
		if isinstance(other, OrderedSet):
			return len(self) == len(other) and list(self) == list(other)
		return set(self) == set(other)

	def __del__(self):
		self.clear()					# remove circular references


#if __name__ == '__main__':
#	 print(OrderedSet('abracadaba'))
#	 print(OrderedSet('simsalabim'))


# ======================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
