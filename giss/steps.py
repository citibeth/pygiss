
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
import giss.util
import time
import shutil

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		pass
#		 if exc.errno == errno.EEXIST and os.path.isdir(path):
#			 pass
#		 else: raise

def remkdir_p(path):
	try:
		shutil.rmtree(path)
	except:
		pass

	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		pass


def rm_f(path):
	try:
		os.remove(path)
	except:
		pass


# def untar(tgz, dest):
# 	mkdir_p(dest)
# 	os.chdir(dest)
# 	tar = tarfile.open(self.tgz_fname)
# 	tar.extractall()
# 	tar.close()
# 
# 	# Find the one directory created
# 	top_untar = None
# 	for leafname in os.listdir(self.src_dir):
# 		fname = os.path.join(self.src_dir, leafname)
# 		if os.path.isdir(fname):
# 			if top_untar is not None:
# 				raise Exception('Untar into {} produced more than one top-level directory!'.format(self.src_dir))
# 			top_untar = fname
# 
# 	# Move stuff up
# 	for fname in os.listdir(top_untar):
# 		shutil.move(os.path.join(top_untar, fname), os.path.join(self.src_dir, fname))





# -------------------------------------------------

class Steps(object):

	def __init__(self, steps_dir):
		self.steps_dir = steps_dir    # Logging and control
		mkdir_p(self.steps_dir)
		self.step_defs = list()
		self.step_defs_by_name = dict()
		self.env = dict(os.environ)
		self.init_fns = list()

	def add_step(self, step_name, step_fn):
		self.step_defs.append((step_name, step_fn))
		self.step_defs_by_name[step_name] = step_fn

	def __iter__(self):
		for step_name, step_fn in self.step_defs:
			yield step_name

	def step_file(self, step_name):
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
		with open(self.step_file(step_name), 'w') as f:
			pass
				
	def run_step(self, stepno, force=False):
		if isinstance(stepno, str):	# It's really a name
			step_name = stepno
			step_fn = self.step_defs_by_name[step_name]
		else:
			step_name, step_fn = self.step_defs[stepno]

		if self.begin_step(step_name):

			print('========== Running {}'.format(step_name))
			try:
				os.remove(self.step_log(step_name))
			except:
				pass


			with open(self.step_log(step_name), 'w') as step_log:
				tee_stdout = giss.util.tee(sys.stdout, step_log)
				tee_stderr = giss.util.tee(sys.stderr, step_log)
				with giss.util.redirect_io(tee_stdout, tee_stderr):
					try:
						step_fn(self)
					except Exception as e:
						tb = traceback.format_exc()
						print(tb)
						raise
		else:
			print('Skipping {}'.format(step_name))

		# We only get here if we succeeded
		# Don't want to run this if we failed above.
		self.end_step(step_name)

		return True

	# ----------------------------------------------------

	def exec(self,cmd, **kwargs):
		print('Running cmd: ', cmd)

		kwargs2 = dict(kwargs)
		if 'env' not in kwargs2:
			kwargs2['env'] = self.env
		if 'shell' not in kwargs2:
			kwargs2['shell'] = False

		proc = subprocess.Popen(cmd,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs2)

		# Join STDOUT and STDERR line-by-line, and send to our STDOUT
		out_prefix = '[o] '
		err_prefix = '[e] '

		last_flush = time.time()
		flush_interval=2.0
		reads = (proc.stdout.fileno(), proc.stderr.fileno())
		while True:
			ret = select.select(reads, [], [], 2.)[0]

			for fd in ret:
				if fd == proc.stdout.fileno():
					read = proc.stdout.readline().decode()
					sys.stdout.write(out_prefix + read)
				elif fd == proc.stderr.fileno():
					read = proc.stderr.readline().decode()
					sys.stdout.write(err_prefix + read)

			now = time.time()
			if (now - last_flush) >= flush_interval:
				sys.stdout.flush()
				last_flush = now

			if proc.poll() != None:
				break

		status = proc.wait()		# We're probably already terminated
		print('status=', status)
		if status != 0:
			raise Exception('Failed on: {}'.format(cmd))

# http://stackoverflow.com/questions/25418499/python-decorators-with-arguments
def step(steps):
	def dec(func):
		steps.add_step(func.__name__, func)
		return func
	return dec

# http://stackoverflow.com/questions/25418499/python-decorators-with-arguments
def init(steps):
	def dec(func):
		steps.init_fns.append(func)
		func(self)
		return func
	return dec
