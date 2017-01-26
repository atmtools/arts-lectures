# -*- coding: utf-8 -*-
#
# Copyright © 2016 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Plot simulated outgoing longwave radiation (OLR) and
Planck curves for different temperatures.
"""
import matplotlib
matplotlib.use('Agg')  # Use Agg-backend to produce plots without X-Server
import matplotlib.pyplot as plt
import numpy as np
import typhon
from typhon.arts import xml
from typhon.plots import styles

# Read ARTS results
olr = xml.load('./results/olr.xml')
f = xml.load('./results/f_grid.xml')
wn = typhon.physics.frequency2wavenumber(f) / 100  # cm^-1
t_surf = float(xml.load('./results/t_surface.xml'))
scene = xml.load('./results/scene.xml')

# Plotting.
plt.style.use(typhon.plots.styles('typhon'))

fig, ax = plt.subplots()
temps = [225, 250, 275, t_surf]
temp_colors = typhon.plots.mpl_colors('temperature', len(temps))
for t, color in sorted(zip(temps, temp_colors)):
    ax.plot(wn, typhon.physics.planck(f, t),
            label='{:3.1f} K'.format(t),
            color=color)
ax.plot(wn, olr, label='Radiance', color='steelblue')
ax.legend(framealpha=0.5)
ax.grid('on')
ax.set_title(r'OLR = {:3.2f} $Wm^{{-2}}Sr^{{-1}}$'.format(np.trapz(olr, f)))
ax.set_xlim(wn.min(), wn.max())
ax.set_xlabel(r'Wave number [$cm^{-1}$]')
ax.set_ylabel(r'Radiance [$Wm^{-2}Hz^{-1}Sr^{-1}$]')
fig.savefig('plots/olr{}.pdf'.format(scene))
