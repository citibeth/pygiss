class EventThrower(object):
	"""Simple class used to manage and call events."""
	def __init__(self):
		self.events = dict()
	def connect(self, eventid, fn):
		if eventid in self.events:
			self.events[eventid].append(fn)
		else:
			self.events[eventid] = [fn]

	def run_event(self, eventid, *args):
		if eventid in self.events:
			for fn in self.events[eventid]: fn(eventid, *args)

	def unconnect(self, eventid, fn):
		self.events.remove(fn)

