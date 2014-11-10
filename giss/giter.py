
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

	def next(self):
		try:
			self.val = next(self.ii)
			return self
		except:
			return None

	def close(self):
		pass

class ChainedGiter(Giter):
	slot: _parent

	def __init__(self, parent):
		"""parent: iterable, or Giter, or a scalar"""
		if isinstance(parent, Giter):
			self._parent = parent
		else:
			if isinstance(parent, collections.Iterable):
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
			return None
		if not self._parent.next():
			return None
		if not self._init_this(start_time, _parent.val):
			reuturn None
		return self

	def next(self):
		while True:
			if self.next_this() is not None: return self
			while True:
				if self._parent.next() is None: return None
				if self._init_this(start_time, _parent.val) is not None: break

	def close(self):
		self.close_this()
		self._parent.close()

class DirGiter(ChainedGiter):
