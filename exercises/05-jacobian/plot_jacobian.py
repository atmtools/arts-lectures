# -*- coding: utf-8 -*-
#
# Based on a script authored by Felix Erdmann
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
p_grid = xml.load('./results/p_grid.xml')
tau = xml.load('./results/optical_thickness.xml')
bt = xml.load('./results/y.xml')
jac = xml.load('./results/jacobian.xml')
alt = xml.load('./results/z_field.xml')

# Find emission level based on "opcaity rule".
z_chapman = alt[np.argmax(tau >= 1.0, axis=0)]

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
plt.semilogy(freq / 10**9, tau[-1, :])
plt.axhline(1, color='black', linewidth=0.8, linestyle='dashed', zorder=-1)
plt.xlabel('Frequency [ GHz ]')
plt.ylabel('Zenith Opacity [  ]')
plt.title('Zenith Opacity')

if freq_ind > 0:
    plt.plot(freq[freq_ind] / 10**9, tau[-1, freq_ind], 'or')

if freq_ind > 0:
    plt.subplot(223)

    plt.plot(jac[freq_ind, :], alt[:, 0, 0] / 1000.0)
    plt.xlim([-1, 7])
    plt.ylim([0, 70])
    plt.xlabel('Jacobian [ K / 1 ]')
    plt.ylabel('Altitude [ km ]')
    plt.title('Water Vapor Jacobian')
    jac_peak = alt[np.argmin(jac[freq_ind, :]), 0, 0].round(1) / 1000
    plt.text(4, 60, f'{jac_peak:.2f} km')
    plt.axhline(jac_peak, color='red', linestyle='dashed', linewidth=0.8)

    opac_layer_space = tau[:, [freq_ind]]

    plt.subplot(224)
    plt.semilogx(opac_layer_space[:, 0], alt[::-1, 0, 0] / 1000.0)
    plt.xlim([10**-13, 10**3])
    plt.xticks([10.**-13, 10.**-10, 10.**-5, 10.**0, 10.**2])
    plt.ylim([0, 70])
    plt.xlabel(r'$\sf Opacity_{layer-space}$ []')
    plt.ylabel('Altitude [ km ]')
    plt.title('Opacity from a layer to space')
    try:
        tau1 = alt[::-1][np.where(opac_layer_space[:, 0] >= 1)[0][0], 0, 0]
    except Exception:
        print('no opacity greater than 1')
    else:
        tau1 /= 1000
        plt.text(0.01, 60, f'{tau1:.2f} km')
        plt.axhline(tau1, color='red', linestyle='dashed', linewidth=0.8)
        plt.axvline(1, ymax=0.2, color='black', linewidth=0.8, zorder=-1)

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
