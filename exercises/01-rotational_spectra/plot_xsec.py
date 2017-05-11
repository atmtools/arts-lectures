# -*- coding: utf-8 -*-
#
# Copyright © 2016 Lukas Kluft <lukas.kluft@gmail.com>
#
# Distributed under terms of the MIT license.

"""Load and plot absorption cross sections corresponding
to arts output from "absorption.arts".
"""
import matplotlib
matplotlib.use('Agg')  # Use Agg-backend to produce plots without X-Server
import matplotlib.pyplot as plt
from typhon.arts import xml
from typhon.plots import styles


# Load the ARTS results.
abs_xsec_all = xml.load('results/abs_xsec_per_species.xml')
abs_xsec = abs_xsec_all[0]
freq = xml.load('results/f_grid.xml')
species = xml.load('results/species.xml')
mol_name = species[0][0].replace('-*', '')
temperature = xml.load('results/temp.xml')
pressure = xml.load('results/press.xml')

# Store setup information in dict.
kwargs = {
    'n': mol_name,
    'p': pressure / 1e2,
    'T': temperature,
    }

# Plot the results.
plt.style.use(styles('typhon'))

fig, ax = plt.subplots()
ax.plot(freq / 1e9, abs_xsec)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Abs. cross section [$m^2$]')
ax.set_title('{n} p:{p:0.0f} hPa T:{T:0.0f} K'.format(**kwargs))

# Save figure.
fig.savefig('plots/plot_xsec_{n}_{p:0.0f}hPa_{T:.0f}K.pdf'.format(**kwargs))