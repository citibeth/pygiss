
# Copy a netCDF file (so we can add more stuff to it)
class copy_nc :
	def __init__(self, nc0, ncout) :
		self.nc0 = nc0
		self.ncout = ncout
		self.vars = nc0.variables.keys()

		# Figure out which dimensions to copy
		copy_dims = set()
		for var in self.vars :
			for dim in nc0.variables[var].dimensions :
				copy_dims.add(dim)

		# Copy the dimensions!
		for dim_pair in nc0.dimensions.items() :
			name = dim_pair[0]
			extent = len(dim_pair[1])
			if name in copy_dims :
				ncout.createDimension(name, extent)

		# Define the variables
		for var_name in self.vars :
			var = nc0.variables[var_name]
			varout = ncout.createVariable(var_name, var.dtype, var.dimensions)
			for aname, aval in var.__dict__.items() :
				setattr(varout, aname, aval)

	def copy_data(self) :
		# Copy the variables
		for var_name in self.vars :
			ivar = self.nc0.variables[var_name]
			ovar = self.ncout.variables[var_name]
			ovar[:] = ivar[:]

