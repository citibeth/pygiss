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

import subprocess
import re

_lineRE = re.compile(r'(.*?)=(.*)\n')

def read_env(fname = None) :
	"""Parses a Bash file full of variable settings by running it and
	seeing what happens.

	Returns:	{string : string}
		Dictionary of the name/value pairs found in the file."""

	cmd = '. %s; set' % fname
	pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
	ret = {}
	for line in pipe :
		match = _lineRE.match(line)
		if match is not None :
			val = match.group(2)
			if len(val) > 0 :
				if (val[0] == "'" and val[-1] == "'") or (val[0] == '"' and val[-1] == '"') :
					val = val[1:-1]

				ret[match.group(1)] = val

	return ret
