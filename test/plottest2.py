# Demonstrate more than one figure on a page.

import netCDF4
import giss.basemap
import giss.modele
import matplotlib.pyplot


nc = netCDF4.Dataset('data/ANN1950.aijhctest45_lr05.nc')

# Use a custom basemap
basemap = giss.basemap.greenland_laea()

# Plot multiple plots on one page
figure = matplotlib.pyplot.figure(figsize=(11,8.5))

ax = figure.add_subplot(121)
pp = giss.modele.plot_params('pr_lndice', nc=nc)
print ax
giss.plot.plot_var(ax=ax, basemap=basemap, **pp)

ax = figure.add_subplot(122)
pp = giss.modele.plot_params('evap', nc=nc)
giss.plot.plot_var(ax=ax, basemap=basemap, **pp)

# Save to a file as png
figure.savefig('fig.png', dpi=300, transparent=True)

# Also show on screen
matplotlib.pyplot.show()
