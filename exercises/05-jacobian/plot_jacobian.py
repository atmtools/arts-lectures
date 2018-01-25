# -*- coding: utf-8 -*-
#
# Author: Felix Erdmann
"""This script uses the output variables of the arts controlfile
`jacobian.arts`. it plots nadir brightness temperature at the top of
the atmosphere, zenith opacity, the water vapour jacobian and the
opacity between altitude z and the top of the atmosphere. Different
frequencies can be selected by changing the variable `freq_ind` into
a number between 1 and 110.
"""
import matplotlib.pyplot as plt
from typhon.arts import xml
import numpy as np

# select frequency
freq_ind = -1

# read in everything
freq = xml.load('./results/f_grid.xml')
iy_aux = xml.load('./results/iy_aux.xml')
opacity = xml.load('./results/y_aux.xml')
bt = xml.load('./results/bt.xml')
jac = xml.load('./results/jacobian.xml')
alt = xml.load('./results/z_field.xml')
mol_name = xml.load('./results/species.xml')

press = iy_aux[0]
abs_ = iy_aux[1]
opacity = opacity[0]

# calculate the level where opacity becomes one
nlevls = len(alt)
nfreqs = len(freq)

# flipping the altitude dimension to start the integration from TOA
abs_ = abs_[:, ::-1]
z_fine = alt[::-1, :]
z_chapmann = np.zeros((nfreqs, 1))

for ifreq in range(nfreqs):
    tau = 0.0
    abs_per_freq = np.squeeze(abs_[ifreq, :])
    for ilev in range(nlevls - 1):
        abs_layer = 0.5 * (abs_per_freq[ilev] + abs_per_freq[ilev + 1])
        tau = tau + abs_layer * (z_fine[ilev] - z_fine[ilev + 1])
        if tau > 1:
            z_chapmann[ifreq, 0] = z_fine[ilev] / 1000.0
            break

# plotting ...
plt.figure(1)
if freq_ind > 0:
    plt.subplot(221)
else:
    plt.subplot(211)
plt.plot(freq / 10**9, bt)
plt.xlabel('Frequency [ GHz ]')
plt.ylabel(r'T$_B$ [ K ]')
plt.title('Brightness Temperature')

if freq_ind > 0:
    plt.plot(freq[freq_ind] / 10**9, bt[freq_ind], 'or')
    plt.text(152, 280, str((freq[freq_ind]/10**9).round(3))+'GHz')

if freq_ind > 0:
    plt.subplot(222)
else:
    plt.subplot(212)
plt.semilogy(freq / 10**9, opacity)
plt.xlabel('Frequency [ GHz ]')
plt.ylabel('Zenith Opacity [  ]')
plt.title('Zenith Opacity')

if freq_ind > 0:
    plt.plot(freq[freq_ind] / 10**9, opacity[freq_ind], 'or')

if freq_ind > 0:
    plt.subplot(223)

    plt.plot(jac[freq_ind, :], alt[:, 0, 0] / 1000.0)
    plt.xlim([-1, 7])
    plt.ylim([0, 70])
    plt.xlabel('Jacobian [ K / 1 ]')
    plt.ylabel('Altitude [ km ]')
    plt.title('Water Vapor Jacobian')
    plt.text(4, 60, str(
        alt[np.argmin(jac[freq_ind, :]), 0, 0].round(1) / 1000) + 'km')

    # Calculate opacity between each level and TOA
    opac_layer_space = np.zeros((nlevls, 1))
    abs_per_freq = np.squeeze(abs_[freq_ind, :])

    tau = 0
    for ilev in range(nlevls - 1):
        abs_layer = 0.5 * (abs_per_freq[ilev] + abs_per_freq[ilev + 1])
        tau = tau + abs_layer * (z_fine[ilev] - z_fine[ilev + 1])
        opac_layer_space[ilev + 1, 0] = tau

    plt.subplot(224)
    plt.semilogx(opac_layer_space[:, 0], z_fine[:, 0, 0] / 1000.0)
    plt.xlim([10**-13, 10**3])
    plt.xticks([10.**-13, 10.**-10, 10.**-5, 10.**0, 10.**2])
    plt.ylim([0, 70])
    plt.xlabel(r'Opacity$_{layer--space}$ [  ]')
    plt.ylabel('Altitude [ km ]')
    plt.title('Opacity from a layer to space')
    try:
        plt.text(0.01, 60, str(z_fine[np.where(opac_layer_space[:, 0] >= 1)[0][0], 0, 0].round(1)/1000)+'km')
    except Exception:
        print('no opacity greater than 1')

# save figure
pathname = './plots/'
if freq_ind < 0:
    figname = 'bt_op_part1.pdf'
    plt.tight_layout()
    plt.savefig(pathname + figname)
else:
    figname = 'jac_' + str(np.round((freq[freq_ind] / 10**9))) + 'GHz.pdf'
    plt.tight_layout()
    plt.savefig(pathname+figname)
