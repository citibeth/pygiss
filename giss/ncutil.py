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


# Copy a netCDF file (so we can add more stuff to it)
class copy_nc :
	def __init__(self, nc0, ncout,
		var_filter=lambda x : True,
		attrib_filter = lambda x : True) :
		"""var_filter : function(var_name) -> bool
		    Only copy variables where this filter returns True.
		attrib_filter : function(attrib_name) -> bool
		    Only copy attributes where this filter returns True."""
		self.nc0 = nc0
		self.ncout = ncout
		self.var_filter = var_filter
		self.attrib_filter = attrib_filter
		self.avoid_vars = set()
		self.avoid_dims = set()

	def createDimension(self, dim_name, *args, **kwargs) :
		self.avoid_dims.add(dim_name)
		return self.ncout.createDimension(dim_name, *args, **kwargs)

	def createVariable(self, var_name, *args, **kwargs) :
		self.avoid_vars.add(var_name)
		return self.ncout.createVariable(var_name, *args, **kwargs)

	def define_vars(self) :
		self.vars = self.nc0.variables.keys()

		# Figure out which dimensions to copy
		copy_dims = set()
		for var in self.vars :
			if var in self.avoid_vars : continue
			for dim in self.nc0.variables[var].dimensions :
				copy_dims.add(dim)

		# Copy the dimensions!
		for dim_pair in self.nc0.dimensions.items() :
			name = dim_pair[0]
			extent = len(dim_pair[1])
			if name in copy_dims :
				self.ncout.createDimension(name, extent)

		# Define the variables
		for var_name in self.vars :
			if not self.var_filter(var_name) : continue
			if var_name in self.avoid_vars : continue
			var = self.nc0.variables[var_name]
			varout = self.ncout.createVariable(var_name, var.dtype, var.dimensions)
			for aname, aval in var.__dict__.items() :
				if not self.attrib_filter(aname) : continue
				setattr(varout, aname, aval)

	def copy_data(self) :
		# Copy the variables
		for var_name in self.vars :
			if not (var_name in self.avoid_vars) :
				ivar = self.nc0.variables[var_name]
				ovar = self.ncout.variables[var_name]
				ovar[:] = ivar[:]

