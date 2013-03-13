# Demonstrates various customization features for plotting
# This tries to be a "kitchen sink" of the kinds of customizations you can
# do for one plot, while still using giss.plot.plot_var().

import netCDF4
import giss.basemap
import giss.modele
import numpy as np
import matplotlib.colors

nc = netCDF4.Dataset('data/ANN1950.aijhctest45_lr05.nc')

# --------- Read / compute something complex on the fly
soilfr = nc.variables['soilfr'][:]
runoff_soil = giss.modele.read_ncvar(nc, 'runoff_soil')
runoff_soil[np.isnan(runoff_soil)] = 0.

landicefr = nc.variables['landicefr'][:]
runoff_lndice = giss.modele.read_ncvar(nc, 'runoff_lndice')
runoff_lndice[np.isnan(runoff_lndice)] = 0.

runoff = (runoff_soil * soilfr) + (runoff_lndice * landicefr)

runoff *= (365. * .001)		# Convert to m/yr
# --------------------------------------

pp = giss.modele.plot_params(val=runoff)

pp['title'] = 'Total Runoff (m/yr)'
plot_args = pp['plot_args']
plot_args['vmin'] = 1.e-10
#plot_args['vmax'] = 1.e2
plot_args['cmap'] = giss.plot.cpt('precip2_17lev.cpt').cmap
#plot_args['cmap'] = giss.plot.cpt('nd/pink/Neutral_01.cpt').cmap
plot_args['norm'] = matplotlib.colors.LogNorm()		# Do a log plot
#plot_args['norm'] = giss.plot.AsymmetricNormalize()  # Center on zero

cb_args = pp['cb_args']
cb_args['ticks'] = [1e-10,1e-8,1e-6,1e-4,1e-2,1,100]
cb_args['format'] = '%g'


plt = giss.plot.plot_var(show=False, **pp)

# Super-custom tick labels
tick_labels = ['$10^{-10}$', '1e-8', '$10^{-6}$', '.0001', '.01', 'unity', '100']
plt['colorbar'].ax.set_xticklabels(tick_labels)


# Display the plot
matplotlib.pyplot.show()
