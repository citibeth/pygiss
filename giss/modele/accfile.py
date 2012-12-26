# Pythonizes functionality of scaleacc
#

import giss.util
import re
import datetime
import netCDF4
import operator
import numpy as np
import os
import os.path
from odict import odict

# ==========================================================
def list_acc_files(dir, rundeck='', date0=None, date1=None) :
	"""List acc files in a directory, and sort by date.

	Args:
		dir (string):
			Directory in which to list files
		rundeck (string):
			Name of rundeck of which we're looking for output
			(omit or set to '' if you wish to list files for all rundecks)
		date0 (datetime.date):
			First of the month, for the first month to list
		date1 (datetime.date):
			First of the month, for the (last+1) month to list

	Returns:	[(date, fname), ...]
		List of pairs of (date, fname)

		date (datetime.date):
			Date of the file
		fname (string):
			Name of the file

		If you want two separate lists instead, do:
			dates, fnames = zip(list_acc_files(...))

	See:
		lsacc.py
		http://stackoverflow.com/questions/5917522/unzipping-and-the-operator"""


	rundeckRE = re.compile(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d\d\d\d)\.acc%s.nc' % rundeck)
	lst = []
	for fname in os.listdir(dir) :
		match = rundeckRE.match(fname)
		if match is None: continue
		month = _monthnums[match.group(1)]
       		year = int(match.group(2))
		dt = datetime.date(year,month,1)

		if date0 is not None and dt < date0 : continue
		if date1 is not None and dt >= date1 : continue

		lst.append((dt, os.path.join(dir,fname)))
	lst.sort()

	return lst

def list_deck_files(decks_dir, rundeck, **kwargs) :
	return list_acc_files(os.path.join(decks_dir, rundeck), rundeck, **kwargs)
# ==========================================================
# "From:  1949  DEC  1,  Hr  0      To:  1950  JAN  1, Hr  0  Model-Time:    17520     Dif:  31.00 Days" ;
_fromtoRE = re.compile(r'From:\s+(\d+)\s+([a-zA-Z]+)\s+(\d+),\s+Hr\s+(\d+)\s+To:\s+(\d+)\s+([a-zA-Z]+)\s+(\d+),\s+Hr\s+(\d+)\s+Model-Time:\s+(\d+)\s+.*')

_monthnums = {
	'JAN' : 1,'FEB' : 2,'MAR' : 3,'APR' : 4,'MAY' : 5,'JUN' : 6,
	'JUL' : 7,'AUG' : 8,'SEP' : 9,'OCT' : 10,'NOV' : 11,'DEC' : 12}

def _parse_fromto(sfromto) :
	match = _fromtoRE.match(sfromto)

	year0 = int(match.group(1))
	month0 = _monthnums[match.group(2)]
	day0 = int(match.group(3))
	hour0 = int(match.group(4))

	year1 = int(match.group(5))
	month1 = _monthnums[match.group(6)]
	day1 = int(match.group(7))
	hour1 = int(match.group(8))

	model_time = int(match.group(9))

	return (
		(datetime.date(year0, month0, day0), hour0),
		(datetime.date(year1, month1, day1), hour1))

# -----------------------------------------------------
# @return List of strings
def _get_var_text(nc, vname) :
	ret = []
	var = nc.variables[vname]
	for i in range(0,len(var)) :
		ret.append(''.join(var[i]).strip())
	return ret

def _get_vara_real(nc, vname, srt, cnt) :
	slices = []
	for i in range(0,len(srt)) :
		slices.append(slice(srt[i], srt[i] + cnt[i]))
	return nc.variables[vname][slices]

# -----------------------------------------------------
def get_acc_categories(nc) :
	"""Get the set of all diagnostic categories
	Args:
		nc (netCDF4.Dataset):
			Open netCDF handle for the ACC file.
	Returns:	set(string)
		Set of all diagnostic categories (eg: 'ij', 'ijhc', etc)."""

	dcats = set()
	for vname in nc.variables.iterkeys() :
		if vname.find('_latlon') >= 0 : continue
		if vname.startswith('cdl_') :
			dcats.add(vname[4:])
	return dcats

# -----------------------------------------------------

def acc_fromto(nc) :
	return _parse_fromto(nc.fromto)

# -----------------------------------------------------
_infoRE = re.compile(r'\s*(.*?):(.*?)\s*=\s*"(.*?)"\s*;')
_lonRE = re.compile(r'\s*lon\s=')
_latRE = re.compile(r'\s*lat\s=')
class ScaleAcc :
	def __init__(self, nc, dcat) :
		self.nc = nc
		self.dcat = dcat

		self.dcatvar = nc.variables[dcat]

		# Read list of symbols
		sname_acc = _get_var_text(nc, 'sname_' + dcat)	# List of strings
		self.snames = {}
		for i in range(0,len(sname_acc)) :
			self.snames[sname_acc[i]] = i

		# Read units and names of symbols
		self.cdl = _get_var_text(nc, 'cdl_' + dcat)
		self.varinfo = {}		# varinfo[var][attr] --> val
		for line in self.cdl :
			match = _infoRE.match(line)
			if match is None: continue
			var = match.group(1)
			attr = match.group(2)
			val = match.group(3)
			
			if var in self.varinfo :
				rec = self.varinfo[var]
			else :
				rec = {'sname' : var}
				self.varinfo[var] = rec
			rec[attr] = val

		# Find the size of the dimension along which to split the data
		if 'split_dim' in self.dcatvar.__dict__ :
			self.split_dim = len(self.dcatvar.shape) - self.dcatvar.split_dim	# Reverse dimensions for Python
			kacc = self.dcatvar.shape[self.split_dim]
		else :
			self.split_dim = -1		# No split
			kacc = 1

		# Read acc metadata needed for scaling
		self.scale_acc = self.nc['scale_' + dcat][:]
		if ('denom_' + dcat in nc.variables) :
			self.denom_acc = self.nc['denom_' + dcat][:]
			self.denom_acc -= 1
		else :
			self.denom_acc = np.zeros(kacc, 'i') - 1

		# Read counters by which to divide
		if ('ia_' + dcat in nc.variables) :
			# Which counter to use for each variable
			self.ia_acc = self.nc['ia_' + dcat][:]
			self.ia_acc -= 1

			# Value of those counters
			self.idacc = self.nc['idacc'][:]
		else :
			# Just use counter #0 for all variables
			self.ia_acc = np.array([0],'i')

			# ...and here is the value for the counter
			if ('ntime_' + dcat in nc.variables) :
				# this acc array has a custom counter, store in idacc[0]
				self.idacc = np.ones(1) * self.nc.variables['ntime_' + dcat][0]
			else :
				self.idacc = np.ones(1)	# Not a traditional accumulation (eg min,max)

		# Conveniene to avoid dividing later
		self.by_idacc = 1.0 / self.idacc

	def var_info(self, sname) :
		return self.varinfo[sname]

	# @return List of netCDF dimension names that describe
	# the output of scale_var()
	def scaled_dims(self) :
		sdims = list(dcatvar.dimensions)
		

	# @param k (0-based) index of the acc variable we want to sacle
	def scale_var(self, sname) :
		# Get index of the variable
		k = self.snames[sname]

		# Retrieve and scale this field
		srt = np.zeros(len(self.dcatvar.shape), 'i')
		cnt = np.array(self.dcatvar.shape, 'i')
		out_shape = list(self.dcatvar.shape)
		out_sdims = list(self.dcatvar.dimensions)
		if self.split_dim >= 0 :
			srt[self.split_dim] = k
			cnt[self.split_dim] = 1
			del out_shape[self.split_dim]
			del out_sdims[self.split_dim]
		out_shape = tuple(out_shape)

		xout = _get_vara_real(self.nc, self.dcat, srt, cnt).reshape(out_shape)
		xout *= self.scale_acc[k] * self.by_idacc[self.ia_acc[k]]

		kd = self.denom_acc[k]
		if kd >= 0:
			srt[self.split_dim] = kd
			xden = _get_vara_real(self.nc, dcat, srt, cnt).reshape(out_shape)

			nonzero = (xden != 0)
			xout[nonzero] *= self.idacc[self.ia_acc[kd]] / xden(nonzero)
			xout[xden==0] = np.nan

		info = dict(self.varinfo[sname])
		info['val'] = xout
		info['sdims'] = out_sdims
		#info['rundeck'] = self.rundeck
		#info['smonth'] = self.smonth
		#info['dcat'] = self.dcat
		return giss.util.Struct(info)
