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


# http://python-future.org/compatible_idioms.html
from __future__ import print_function    # (at top of module)

import sys
import os
import tarfile
import shutil
import subprocess
import grp
import pwd
import re
import traceback
import select
from giss import giutil
import time
import shutil


# def untar(tgz, dest):
#   mkdir_p(dest)
#   os.chdir(dest)
#   tar = tarfile.open(self.tgz_fname)
#   tar.extractall()
#   tar.close()
# 
#   # Find the one directory created
#   top_untar = None
#   for leafname in os.listdir(self.src_dir):
#       fname = os.path.join(self.src_dir, leafname)
#       if os.path.isdir(fname):
#           if top_untar is not None:
#               raise Exception('Untar into {} produced more than one top-level directory!'.format(self.src_dir))
#           top_untar = fname
# 
#   # Move stuff up
#   for fname in os.listdir(top_untar):
#       shutil.move(os.path.join(top_untar, fname), os.path.join(self.src_dir, fname))





# -------------------------------------------------

class Steps(object):

    def __init__(self, steps_dir):
        self.steps_dir = steps_dir    # Logging and control
        self.mkdir_p(self.steps_dir)
        self.step_defs = list()
        self.step_defs_by_name = dict()
        self.step_files = dict()
        self.env = dict(os.environ)
        self.init_fns = list()

    def add_step(self, step_name, step_fn, step_file=None):
        self.step_defs.append((step_name, step_fn))
        self.step_defs_by_name[step_name] = step_fn
        if step_file is not None:
            self.step_files[step_name] = step_file

    def __iter__(self):
        for step_name, step_fn in self.step_defs:
            yield step_name

    def step_file(self, step_name):
        try:
            return self.step_files[step_name]
        except:
            return os.path.join(self.steps_dir, '{}'.format(step_name))

    def step_log(self, step_name):
        return os.path.join(self.steps_dir, '{}.log'.format(step_name))

    # ----------------------------------------------------
    def begin_step(self, step_name, force=False):
        step_file = self.step_file(step_name)
        if force:
            try:
                os.remove(step_file)
            except:
                pass
            self.step_name = step_name
            return True

        if not os.path.isfile(step_file):
            self.cur_step = step_name
            return True
        return False

    def end_step(self, step_name, force=False):
        step_file = self.step_file(step_name)
        if not os.path.exists(step_file):
            with open(step_file, 'w') as f:
                pass
                
    def run_step(self, stepno, force=False):
        if isinstance(stepno, str): # It's really a name
            step_name = stepno
            step_fn = self.step_defs_by_name[step_name]
        else:
            step_name, step_fn = self.step_defs[stepno]

        if self.begin_step(step_name):

            print('========== Running {}'.format(self.step_file(step_name)))
            try:
                os.remove(self.step_log(step_name))
            except:
                pass


            with open(self.step_log(step_name), 'w') as step_log:
                tee_stdout = giutil.tee(sys.stdout, step_log)
                tee_stderr = giutil.tee(sys.stderr, step_log)
                with giutil.redirect_io(tee_stdout, tee_stderr):
                    try:
                        step_fn(self)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
                        raise
        else:
            print('Skipping {}'.format(self.step_file(step_name)))

        # We only get here if we succeeded
        # Don't want to run this if we failed above.
        self.end_step(step_name)

        return True

    def run_all(self):
        for step_name in iter(self):
            self.run_step(step_name)


    # ----------------------------------------------------

    def exec(self,cmd, **kwargs):
        print('Running cmd: ', cmd)
        print('Running cmd: ', ' '.join(cmd))

        kwargs2 = dict(kwargs)
        if 'env' not in kwargs2:
            kwargs2['env'] = self.env
        if 'shell' not in kwargs2:
            kwargs2['shell'] = False

        proc = subprocess.Popen(cmd, **kwargs2)
#           stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs2)
#
#       # Join STDOUT and STDERR line-by-line, and send to our STDOUT
#       out_prefix = '[o] '
#       err_prefix = '[e] '
#
#       last_flush = time.time()
#       flush_interval=2.0
#       reads = (proc.stdout.fileno(), proc.stderr.fileno())
#       while True:
#           ret = select.select(reads, [], [], 2.)[0]
#
#           for fd in ret:
#               if fd == proc.stdout.fileno():
#                   read = proc.stdout.readline().decode()
#                   sys.stdout.write(out_prefix + read)
#               elif fd == proc.stderr.fileno():
#                   read = proc.stderr.readline().decode()
#                   sys.stdout.write(err_prefix + read)
#
#           now = time.time()
#           if (now - last_flush) >= flush_interval:
#               sys.stdout.flush()
#               last_flush = now
#
#           if proc.poll() != None:
#               break

        status = proc.wait()        # We're probably already terminated
        print('status=', status)
        if status != 0:
            raise Exception('Failed on: {}'.format(cmd))


    # --------------------------------------------
    def mkdir_p(self, path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            pass
    #        if exc.errno == errno.EEXIST and os.path.isdir(path):
    #            pass
    #        else: raise

    def remkdir_p(self, path):
        try:
            shutil.rmtree(path)
        except:
            pass

        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            pass


    def rm_f(self, path):
        try:
            os.remove(path)
        except:
            pass


                
# http://stackoverflow.com/questions/25418499/python-decorators-with-arguments
def step(steps, **kwargs):
    def dec(func):
        steps.add_step(func.__name__, func, **kwargs)
        return func
    return dec
