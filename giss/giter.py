import collections
import re
import os

class Giter(object):
	def init(self, start_time):
		"""Initializes to start at the specified time.  May be called more
		than once (especially if this Giter has a parent Giter."""
		return None

	def next(self):

		"""Puts the next item in self.val (and maybe other fields as
		well, if convenient).  Returns non-None if successful, None if
		no more items available."""
		return None

	def close(self):
		pass

class IterGiter(Giter):
	def __init__(self, my_iterable):
		self.my_iterable = my_iterable

	def init(self, start_time):
		self.ii = iter(self.my_iterable)
		return self

	def next(self):
		try:
			self.val = next(self.ii)
			return self
		except:
			return None

	def close(self):
		pass

def single_giter(item):
	return IterGiter([item])

class ChainedGiter(Giter):
#	slot: _parent
	__slots__ = ['parent']

	def __init__(self, parent):
		"""parent: iterable, or Giter, or a scalar"""
		if isinstance(parent, Giter):
			self._parent = parent
		else:
			if isinstance(parent, str):
				# Just us it as a single-value scalar
				self._parent = IterGiter([parent])
			elif isinstance(parent, collections.Iterable):
				# First see if this is iterable
				self._parent = IterGiter(parent)
			else:
				# Just us it as a single-value scalar
				self._parent = IterGiter([parent])

	# ----------------------------------------------
	def init_this(self, start_time, parent_val):
		return None
	def next_this(self):
		return None
	def close_this(self):
		pass

	# ----------------------------------------------
	def init(self, start_time):
		if not self._parent.init(start_time):
			print self._parent
			return None
		if not self._parent.next():
			return None
		if not self.init_this(start_time, self._parent.val):
			return None
		return self

	def next(self):
		while True:
			if self.next_this() is not None: return self
			while True:
				if self._parent.next() is None: return None
				if self._init_this(start_time, self._parent.val) is not None: break

	def close(self):
		self.close_this()
		self._parent.close()

DirGiterVal = collections.namedtuple('DirGiterVal', ['fname', 'leafname'])

class DirGiter(ChainedGiter):
	def __init__(self, parent, fileRE):
		super(DirGiter,self).__init__(parent)
		self.fileRE = fileRE

	def init_this(self, start_time, parent_val):
		self.dir = parent_val
		self.dir_list = iter(os.listdir(parent_val))
		return self

	def next_this(self):
		while True:
			try:
				fname = next(self.dir_list)
				match = self.fileRE.match(fname)
				if match is not None:
					self.val = os.path.join(self.dir, fname)
					return self
			except StopIteration:
				return None

def iterator(giter, *args, **kwargs):
	if giter is None:
		return
	if giter.init(*args, **kwargs) is None:
		return
	while giter.next() is not None:
		yield giter.val

	