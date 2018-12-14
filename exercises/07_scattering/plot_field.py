# -*- coding: utf-8 -*-
#
# Copyright © 208 Jakob Doerr <jakobdoerr@googlemail.com>
#
#
"""
Load and plot radiation fields from the DOIT scattering solver, corresponding
to the output of scattering.arts.
"""
import numpy as np
import matplotlib
#matplotlib.use('Agg')  # Use Agg-backend to produce plots without X-Server
import matplotlib.pyplot as plt
from typhon.arts import xml
from typhon.plots import styles
from typhon import physics

plt.close('all')
plt.style.use(styles('typhon'))
zenith_angle = -1
pressure_level = -1

## Mass of the scattering particle
mass = 1.295954e-8

f = xml.load('./output/f_grid.xml')
p = xml.load('./output/p_grid.xml')
t = xml.load('./output/t_field.xml')
pnd = xml.load('./output/pnd_field.xml')

cloudbox_limits = xml.load('./output/cloudbox_limits.xml')
zenith_angles = xml.load('./output/za_grid.xml')
p = p[cloudbox_limits[0]:cloudbox_limits[1]+1]
t = t[cloudbox_limits[0]:cloudbox_limits[1]+1]
mass_mixing_ratio = mass*pnd[0,cloudbox_limits[0]:cloudbox_limits[1]+1,0,0]

ifield = xml.load('./output/DOIT_ifield.xml')
ifield_clearsky = xml.load('./output/DOIT_ifield_clearsky.xml')
ifield = np.squeeze(ifield)
ifield_clearsky = np.squeeze(ifield_clearsky)
ifield = physics.radiance2planckTb(f,ifield)
ifield_clearsky = physics.radiance2planckTb(f,ifield_clearsky)

## Plot Tb vs height for a specific viewing angle
f0, a0 = plt.subplots(1,1 ,figsize=(10,9))
a0.plot(ifield[:,zenith_angle],p/100)
a0.plot(ifield_clearsky[:,zenith_angle],p/100)
a0.set_xlim([np.min(ifield[:,zenith_angle])-10,np.max(ifield[:,zenith_angle])+10])
a0.grid()
a0.invert_yaxis()
a0.set_ylabel('Pressure [hPa]')
a0.set_xlabel('Brightness Temperature [K]')
a0.legend(['Scattering','Clear-sky'])
a0.set_title(r'Brightness Temperature at $\Theta$ = %d °'%(zenith_angles[zenith_angle]))
f0.savefig('./plots/Brightness_temperature_angle_%d.pdf'%(zenith_angles[zenith_angle]))

## Plot Tb vs Viewing angle for a specific pressure level:
if pressure_level > -1:
	f1, a1 = plt.subplots(1,1, figsize=(10,9))
	a1.plot(zenith_angles,ifield[pressure_level,:])
	a1.plot(zenith_angles,ifield_clearsky[pressure_level,:])
	a1.set_ylim([np.min(ifield_clearsky.data[pressure_level,:])-10,np.max(ifield_clearsky.data[pressure_level,:])+10])
	a1.grid()
	a1.set_xlabel('Viewing angle [°]')
	a1.set_ylabel('Brightness Temperature [K]')
	a1.legend(['Scattering','Clear-sky'])
	a1.set_title('Brightness Temperature at p = %d hPa'%(p[pressure_level]/100))
	f1.savefig('./plots/Brightness_temperature_pressure_%dhPa.pdf'%(p[pressure_level]/100))
	print('Temperature at %dhPa: %dK'%(p[pressure_level]/100,t[pressure_level]))
