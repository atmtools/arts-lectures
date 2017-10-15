# -*- coding: utf-8 -*-
#
# Copyright © 2017 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Load and plot zenith opacity and brightness temperatures.
"""
import matplotlib
matplotlib.use('Agg')  # Use Agg-backend to produce plots without X-Server
import matplotlib.pyplot as plt
from typhon.arts import xml
from typhon.plots import styles


plt.style.use(styles('typhon'))


def trim_speciestags(speciestags):
    """Return trimmed and flat version of ArrayOfSpeciesTags."""
    return [s[0].split('-')[0] for s in speciestags]


# Read ARTS results
bt = xml.load('results/bt.xml')
freq = xml.load('results/f_grid.xml')
od = xml.load('results/odepth_1D.xml')
species = trim_speciestags(xml.load('results/species.xml'))
height = int(xml.load('results/sensor_pos.xml'))
zenith_angle = int(xml.load('results/sensor_los.xml'))

# figure of zenith opacity with logarithmic scale on y axis
fig, ax = plt.subplots()
ax.semilogy(freq / 1e9, od)
ax.grid(True)
ax.set_xlim(freq.min() / 1e9, freq.max() / 1e9)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Zenith opacity')
ax.set_title('{s}, {h}km, {z}°'.format(
    s=', '.join(species), h=height, z=zenith_angle))
fig.savefig('plots/opacity_{s}_{h}km_{z}deg.pdf'.format(
    s='+'.join(species), h=height / 1e3, z=zenith_angle))

# figure of brithtness temperature for defined sensor position and line of
fig, ax = plt.subplots()
ax.plot(freq / 1e9, bt)
ax.grid(True)
ax.set_xlim(freq.min() / 1e9, freq.max() / 1e9)
ax.set_ylim(bottom=0)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Brightness temperature [K]')
ax.set_title('{s}, {h}km, {z}°'.format(
    s=', '.join(species), h=height, z=zenith_angle))
fig.savefig('plots/brightness_temperature_{s}_{h}km_{z}deg.pdf'.format(
    s='+'.join(species), h=height / 1e3, z=zenith_angle))
