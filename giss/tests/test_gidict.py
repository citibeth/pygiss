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

import tempfile
import unittest
from giss import memoize,checksum,gidict
import os
import shutil

class TestBDBDict(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.bdb_fname = os.path.join(self.dir, 'test.db')
        self.bdb = gidict.bdbdict(self.bdb_fname)

    def tearDown(self):
        self.bdb.close()
        shutil.rmtree(self.dir)

    def test_bdb(self):
        self.bdb['hello'] = 'world'
        self.bdb[('yp', 17)] = 'yellow pig'
        self.assertEqual('world', self.bdb['hello'])
        self.assertEqual('yellow pig', self.bdb[('yp', 17)])



if __name__ == '__main__':
    unittest.main()
