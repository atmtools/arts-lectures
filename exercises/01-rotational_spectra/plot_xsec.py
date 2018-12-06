# -*- coding: utf-8 -*-
"""Load and plot absorption cross sections corresponding
to arts output from "absorption.arts".
"""
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
ax.set_xlim(freq.min() / 1e9, freq.max() / 1e9)
ax.set_ylim(bottom=0)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Abs. cross section [$\sf m^2$]')
ax.set_title('{n} p:{p} hPa T:{T:0.0f} K'.format(**kwargs))

# Save figure.
fig.savefig('plots/plot_xsec_{n}_{p}hPa_{T:.0f}K.pdf'.format(**kwargs))
