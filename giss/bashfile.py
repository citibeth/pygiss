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
