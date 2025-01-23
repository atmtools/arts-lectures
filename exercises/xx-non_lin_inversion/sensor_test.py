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


sensor_description, NeDT, Accuracy, FWHM_Antenna=nlo.Hamp_channels(['K','V','W','F','G'], rel_mandatory_grid_spacing=1./1.)

ws = nlo.basic_setup([],sensor_description=sensor_description)

print(len(ws.f_grid.value))

nlo.set_sensor_position_and_view(ws, sensor_altitude, sensor_los)

y_coarse, jacobian = nlo.forward_model(
    ws,
    atm,
    surface_reflectivity,
    surface_temperature,
    retrieval_quantity="H2O",
)


# =============================================================================
# %%
# =============================================================================


sensor_description, NeDT, Accuracy, FWHM_Antenna=nlo.Hamp_channels(['K','V','W','F','G'], rel_mandatory_grid_spacing=1./60.)

ws = nlo.basic_setup([],sensor_description=sensor_description)

print(len(ws.f_grid.value))

nlo.set_sensor_position_and_view(ws, sensor_altitude, sensor_los)

y, jacobian = nlo.forward_model(
    ws,
    atm,
    surface_reflectivity,
    surface_temperature,
    retrieval_quantity="H2O",
)

