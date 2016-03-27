# PyGISS: Misc. Python library
# Copyright (c) 2013-2016 by Elizabeth Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

