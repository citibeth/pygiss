# See....
import os
import sys
import subprocess
import select
import pickle
from giss import giutil
import time
import traceback
import gzip
import io
import importlib

def server():
    """Run the server.  This is to be called via
    ssh and the thunkserver script."""
    bstdin = sys.stdin.buffer   # Ready binary stdin stream
    bstdout = sys.stdout.buffer

    bstdout.write(b'Starting Thunk Server\n')
    context = dict()
    while True:
        result = dict()
        bstdout.flush()
        thunk = pickle.load(bstdin)
        try:
            ret = thunk(context)
            result['ret'] = ret
        except Exception as e:
            sys.stdout.flush()
            sys.stderr.flush()
            tb = traceback.format_exc()
            result['traceback'] = tb
            result['exception'] = e

        bstdout.write(b'BEGIN RESULT\n')
        gz = gzip.GzipFile(fileobj=bstdout)
#       gz = bstdout
        pickle.dump(result, gz)
        gz.close()
        bstdout.write(b'\nEND RESULT\n')
        bstdout.flush()

# =======================================================

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



class Client(object):
    OPEN = 0
    INRESULT = 1
    TERMINATED = 2

    def __init__(self, host, cwd_request=None):

        # ---------- Set up initial configuration

        # Get the CWD on the local machine, in case we want it.
        # Convert to tilde notation, in case home directory is at a differet
        # place on the remote system.
        cwd_client = os.path.join('~', os.path.relpath(os.getcwd(), os.environ['HOME']))
        cwd_server = cwd_client if cwd_request is None else cw_request
        if cwd_request is None:
            cwd_request = '.'

        hosts = dict()
        hosts['localhost'] = dict(cmd = ['thunkserver'])
        hosts[''] = dict(\
            cmd = ['ssh', '-XY', host, 'source', '~/.profile', ';', 'cd', cwd_request, ';', 'thunkserver'])

        config = dict()
        config['host'] = host
        config['cwd_request'] = cwd_request
        config['cwd_client'] = cwd_client
        config['cwd_server'] = cwd_server
        config['hosts'] = hosts


        # ----- Modify config via ~/.thunkserverrc
        HOME = os.environ['HOME']
        fname = os.path.join(HOME, '.thunkserverrc')
        try:
            giutil.read_config(fname, config)
        except Exception as e:
            sys.stderr.write('WARNING: Could not read {}\n    {}\n'.format(fname, e))

        # ----- Get thunkserver command and open pipe to thunkserver
        hosts = config['hosts']
        hdict = hosts[host] if host in hosts else hosts['']
        cmd = hdict['cmd']
        print('Thunkserver cmd = {}'.format(cmd))

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
                                sys.stdout.buffer.write(b'[o] ')
                                sys.stdout.buffer.write(line)
                                sys.stdout.buffer.write(b'\n')

                        elif self.state == Client.INRESULT:
                            if line == b'END RESULT':
                                self.state = Client.OPEN
                                result_len = sum([len(x) for x in self.result_lines]) + len(self.result_lines) - 1
                                sys.stderr.write('Result has size {}\n'.format(result_len))


                                result_s = b'\n'.join(self.result_lines)
                                result_io = io.BytesIO(result_s)
                                gz = gzip.GzipFile(fileobj=result_io)
#                               gz = result_io
                                result = pickle.load(gz)
                            else:
                                self.result_lines.append(line)
                else:
                    for line in lines:
                        sys.stderr.buffer.write(b'[e] ')
                        sys.stderr.buffer.write(line)
                        sys.stderr.buffer.write(b'\n')

            sys.stdout.buffer.flush()
            sys.stderr.buffer.flush()

            if result is not None:
                return result



    def exec(self, thunk):
        result = self._exec(thunk)
        if 'exception' in result:
            if 'traceback' in result:
                sys.stderr.write(result['traceback'])
                sys.stderr.write('\n')
            sys.stdout.flush()
            sys.stderr.flush()
            raise result['exception']
        else:
            return result['ret']

# -------------------------------------------------
class ObjThunk(object):
    """For thunking a bound function on a thunkserver --- where
    the object being called is stored on the thunkserver in a
    context dict."""
    def __init__(self, vname, fn_name, *args, **kwargs):
        self.vname = vname
        self.fn_name = fn_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, con, *myargs, **mykwargs):
        obj = con[self.vname]

        # Combine bound with new arguments
        args = myargs + self.args
        kwargs = dict(self.kwargs)
        for k,v in mykwargs: kwargs[k] = v

        fn = getattr(obj, self.fn_name)
        sys.stdout.flush()
        return fn(*args, **kwargs)
# -------------------------------------------------
class ImportThunk(object):
    """Special kind of thunk to import libraries.  This is used
    to get maptlotlib on the right foot with GTK."""
    def __init__(self, imports):
        self.imports = imports
    def __call__(self, con):
        for smod in self.imports:
            importlib.import_module(smod)


# ===============================================
