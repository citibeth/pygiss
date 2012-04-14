import numpy as np
import snowdrift

nx1 = 8;
ny1 = 8;
nx2 = 32;
ny2 = 32;

ZG = np.zeros((nx1*ny1,), dtype='d')
ZH = np.zeros((nx2*ny2,), dtype='d')
ZG2 = np.zeros((nx1*ny1,), dtype='d')

for ix in range(0,nx1) :
	for iy in range(0,ny1) :
		ZG[iy * nx1 + ix] = (ix+1) * 1.8 + (iy+1)


sd=snowdrift.Snowdrift('xy_overlap.nc')
print 'Python calling sd.downgrid()'

print type(ZG)


print ZG

sd.downgrid(ZG, ZH)
print ZH


sd.upgrid(ZH, ZG2)
print ZG2
