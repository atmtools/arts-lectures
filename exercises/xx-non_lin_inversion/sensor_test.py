#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:44:20 2025

@author: u242031
"""
import numpy as np
import matplotlib.pyplot as plt

import pyarts as pa
import nonlin_oem as nlo


# =============================================================================
# %% read data
# =============================================================================

# surface reflectivity
# It is assumed that the surface reflectivity is known and constant
surface_reflectivity = 0.4

# surface temperature
# It is assumed that the surface temperature is known and constant
surface_temperature = 300.0  # K

# define sensor positions and line of sight
# we assume a HALO like airplane with a sensor at 15 km altitude and a line of sight of 180 degrees
sensor_altitude = 15000.
sensor_los = 180.

# load true data
atms = pa.xml.load("atmosphere/atmospheres_true.xml")
atm = atms[226]

# =============================================================================
# %%
# =============================================================================


sensor_description_coarse, NeDT, Accuracy, FWHM_Antenna=nlo.Hamp_channels(['K','V','W','F','G'], rel_mandatory_grid_spacing=1./1.)

y_coarse, _ = nlo.Forward_model(
    [],
    atm,
    surface_reflectivity,
    surface_temperature,
    sensor_altitude,
    sensor_los,
    sensor_description=sensor_description_coarse
)

# =============================================================================
# %%
# =============================================================================


sensor_description_fine, NeDT, Accuracy, FWHM_Antenna=nlo.Hamp_channels(['K','V','W','F','G'], rel_mandatory_grid_spacing=1./60.)


y_fine, _= nlo.Forward_model(
    [],
    atm,
    surface_reflectivity,
    surface_temperature,
    sensor_altitude,
    sensor_los,
    sensor_description=sensor_description_fine
)


# =============================================================================
# %%
# =============================================================================

# Create subplot grid
bands = ['K','V','W','F','G']



# Create subplot grid with band definitions
fig2, axs2 = plt.subplots(len(bands), 1, figsize=(10, 3*len(bands)), sharex=False)

for i, band in enumerate(bands):

    # Get band definitions from Hamp_channels
    sensor_description_ref, _, _, _ = nlo.Hamp_channels([band])

    print(band)

    freqs_ref=sensor_description_ref[:,0]+sensor_description_ref[:,1]+sensor_description_ref[:,2]
    freqs=sensor_description_fine[:,0]+sensor_description_fine[:,1]+sensor_description_fine[:,2]

    band_idx=[]
    for j, f in enumerate(freqs_ref):
        for k, g in enumerate(freqs):
            if f==g:
                band_idx.append(k)


    # Plot data for current band

    axs2[i].plot(freqs_ref, y_fine[band_idx], 'x-', label='high res',linewidth=2, markersize=10)
    axs2[i].plot(freqs_ref, y_coarse[band_idx], '+', label='low res', markersize=10)

    # Customize plot
    axs2[i].legend()
    axs2[i].set_ylabel('TB [K]')
    axs2[i].grid(True)
    axs2[i].set_title(f'Band {band}')
    axs2[i].set_xticks(freqs_ref)
    axs2[i].set_xticklabels([f'{x/1e9:.2f}' for x in freqs_ref], rotation=45)

# Set common x-axis labels
plt.xlabel('Frequency [GHz]')
plt.tight_layout()

