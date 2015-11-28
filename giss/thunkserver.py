# See....

import os
import sys
import subprocess
import select
import pickle
import giss.util as giutil
import time

# http://stackoverflow.com/questions/5486717/python-select-doesnt-signal-all-input-from-pipe
class LineReader(object):

    def __init__(self, fd):
        self._fd = fd
        self._buf = b''

    def fileno(self):
        return self._fd

    def readlines(self):
        data = os.read(self._fd, 4096)
        if not data:
            # EOF
            return None
        self._buf += data
        if b'\n' not in data:
            return []
        tmp = self._buf.split(b'\n')
        lines, self._buf = tmp[:-1], tmp[-1]
        return lines




def server():

	bstdin = sys.stdin.buffer   # Ready binary stdin stream
	bstdout = sys.stdout.buffer

	bstdout.write(b'Starting Thunk Server\n')
	while True:
		result = dict()
		bstdout.flush()
		thunk = pickle.load(bstdin)
		bstdout.write(b'AA2\n')
		try:
			ret = thunk()
			bstdout.write('ret = {}\n'.format(ret).encode())
			result['ret'] = ret
		except Exception as e:
			result['exception'] = e

		bstdout.write(b'Hello World\n')
		bstdout.write(b'BEGIN RESULT\n')
		pickle.dump(result, bstdout)
		bstdout.write(b'\nEND RESULT\n')
		bstdout.flush()

class Client(object):
	OPEN = 0
	INRESULT = 1
	TERMINATED = 2

	def __init__(self):
		# Convert to tilde notation, in case home directory is at a differet
		# place on the remote system.
		cwd = os.path.join('~', os.path.relpath(os.getcwd(), os.environ['HOME']))

		cmd = ['ssh', '-XY', 'gibbs', 'source', '~/.profile', ';', 'cd', cwd, ';', 'thunkserver']

		self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		self.state = Client.OPEN

		self.stdout_lr = LineReader(self.proc.stdout.fileno())
		self.readable = [ self.stdout_lr,
			LineReader(self.proc.stderr.fileno())]
		self.result_lines = list()

	def _exec(self, thunk):
		if self.state == Client.TERMINATED:
			return {'exception': IOError('Remote server terminated.')}

		p = self.proc

		# Send the thunk
		print('Sending thunk:', thunk)
		pickle.dump(thunk, p.stdin)
		p.stdin.flush()

		# Look for the Thunk result in stdout
		result = None
		while True:
			ready = select.select(self.readable, [], [])[0]
			if not ready:
				continue

			for stream in ready:
				lines = stream.readlines()
				if lines is None:
					# EOF on this stream
					self.readable.remove(stream)
					continue
				if stream == self.stdout_lr:
					result = None
					for line in lines:
						if self.state == Client.OPEN:
							if line == b'BEGIN RESULT':
								self.state = Client.INRESULT
								self.result_lines.clear()
							else:
								sys.stdout.buffer.write(line)
								sys.stdout.buffer.write(b'\n')

						elif self.state == Client.INRESULT:
							if line == b'END RESULT':
								self.state = Client.OPEN
								result = pickle.loads(b'\n'.join(self.result_lines))
							else:
								self.result_lines.append(line)
				else:
					for line in lines:
						sys.stderr.buffer.write(line)

			sys.stdout.buffer.flush()
			sys.stderr.buffer.flush()

			if result is not None:
				return result



	def exec(self, thunk):
		result = self._exec(thunk)
		print('result', result)
		if 'exception' in result:
			raise result['exception'] from None
		else:
			return result['ret']

# -------------------------------------------------
def sample_fn(x):
	print('thunk hello world', x+17)
	sys.stdout.flush()
	return x+17
