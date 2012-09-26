# A template to customize
def plot_global_tpl(ax, var, grid) :
	# create Basemap instance for Robinson projection.
	# coastlines not used, so resolution set to None to skip
	# continent processing (this speeds things up a bit)
	mymap = mpl_toolkits.basemap.Basemap(projection='robin',lon_0=0,resolution='l')

	# draw line around map projection limb.
	# color background of map projection region.
	# missing values over land will show up this color.
	mymap.drawmapboundary(fill_color='0.5')
	mymap.drawcoastlines()

	# Decide on our colormap
	plotargs = {}

	#plotargs['cmap'] = plt.cm.jet
	plotargs['cmap'], vmin, vmax = giss.plotutil.read_cpt_data('cpt-city/grass/precipitation.cpt')
	#plotargs['cmap'] = plt.cm.jet

	# plotargs['vmin'] = vmin
	# plotargs['vmax'] = vmax
	plotargs['shading'] = 'flat'

	# plot our variable
	xx,yy = giss.plotutil.make_mesh(mymap, grid)
	val_masked = giss.modele.mask_acc(var)
	im1 = mymap.pcolormesh(xx, yy, val_masked, plotargs)

	# draw parallels and meridians, but don't bother labelling them.
	mymap.drawparallels(np.arange(-90.,120.,30.))
	mymap.drawmeridians(np.arange(0.,420.,60.))

	# add colorbar
	cb = mymap.colorbar(im1,"bottom", size="5%", pad="2%")

	# Add Title
	if 'title' in kwargs : title = kwargs['title']
	else :
		title = '%s\n%s (%s)' % (var.long_name, var.sname, var.units)
	ax.set_title(title)

	return (mymap, cb)
# --------------------------------------------------------
