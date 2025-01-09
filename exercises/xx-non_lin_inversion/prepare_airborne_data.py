#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 19:53:32 2025

Prepare data for exercise - Non-linear inversion
Read in data from GEM tropical pacific scene and extract
a slice of the data for given latitude and altitude range for
a simulation of an airborne radiometer observation.

To keep it simple the slice will be simply the center in x of the data.

It also creates a dropsonde data set by selecting one of the profiles.

The data will be saved in ARTS XML format.

The data is also plotted for a quick check.

The data is saved in the folder 'observation' in the current directory.

The data is saved in the following files:
- atmospheres_true.xml
- aux1d_true.xml
- aux2d_true.xml
- dropsonde.xml

The plots are saved in the folder 'check_plots' in the current directory.

The plots are saved in the following files:
- profiles.pdf
- columns.pdf
- dropsonde.pdf

@author: Manfred Brath
"""

import os
from copy import deepcopy
import numpy as np
import xarray as xa
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# import seaborn as sns

import pyarts as pa

from typhon.physics import e_eq_mixed_mk
from typhon.constants import gas_constant_water_vapor

# %% paths / constants

data_folder = (
    "/scratch/u237/user_data/mbrath/EarthCARE_Scenes/39320_test_data/"
)

# lat_range = [15.5, 16.5]  # °
lat_range = [16, 17]
alt_max = 15e3  # m

# Amount of Oxygen
O2vmr = 0.2095

# Amount of Nitrogen
N2vmr = 0.7808

#T emperature offset of dropsonde
T_offset=1. #K

# %% load data


file_list = os.listdir(data_folder)

# get rid of non_nc files
files = [file for file in file_list if ".nc" in file]

# estimate common prefix
idx = [
    i for i in range(min(len(files[0]), len(files[1]))) if files[0][i] == files[1][i]
]


# get variable names from list
variable_names = [file[idx[-1] + 1 : -3] for file in files]


# %% variable translation
translator = {
    "water_content_cloud": "scat_species-LWC-mass_density",
    "water_content_ice": "scat_species-IWC-mass_density",
    "water_content_rain": "scat_species-RWC-mass_density",
    "water_content_snow": "scat_species-SWC-mass_density",
    "water_content_graupel": "scat_species-GWC-mass_density",
    "water_content_hail": "scat_species-HWC-mass_density",
    "number_concentration_cloud": "scat_species-LWC-number_density",
    "number_concentration_ice": "scat_species-IWC-number_density",
    "number_concentration_rain": "scat_species-RWC-number_density",
    "number_concentration_snow": "scat_species-SWC-number_density",
    "number_concentration_graupel": "scat_species-GWC-number_density",
    "number_concentration_hail": "scat_species-HWC-number_density",
    "relative_humidity": "abs_species-H2O",
    "temperature": "T",
    "height_thermodynamic": "z",
    "pressure_thermodynamic": "p_grid",
}


# Atmospheric field names
atm_fieldnames = [
    "T",
    "z",
    "scat_species-LWC-mass_density",
    "scat_species-IWC-mass_density",
    "scat_species-RWC-mass_density",
    "scat_species-SWC-mass_density",
    "scat_species-GWC-mass_density",
    "scat_species-HWC-mass_density",
    "scat_species-LWC-number_density",
    "scat_species-IWC-number_density",
    "scat_species-RWC-number_density",
    "scat_species-SWC-number_density",
    "scat_species-GWC-number_density",
    "scat_species-HWC-number_density",
    "abs_species-H2O",
    "abs_species-O2",
    "abs_species-N2",
]

# %%

# first read latitude
lat_idx = [i for i in range(len(files)) if "latitude" in files[i]][0]
lat = xa.load_dataarray(data_folder + files[lat_idx]).to_numpy()
central_idx = np.size(lat, axis=0) // 2

latitude = lat[central_idx, :]
lat_range_idx = np.where(
    np.logical_and(latitude > lat_range[0], latitude < lat_range[1])
)[0]


# %% now read data

rawdata = {}
cnt = 0
N_nvars = len(variable_names)

for var in variable_names:

    cnt += 1
    print(f"\n{cnt} of {N_nvars}")
    print(f"reading {var}")

    if var in translator:
        varname = translator[var]
        print(f"rename {var} -> {varname}")
    else:
        varname = var
    idx = [i for i in range(len(files)) if var in files[i]][0]

    temp = xa.open_dataarray(data_folder + files[idx])
    if temp.ndim == 3:
        rawdata[varname] = np.float64(
            temp[:, central_idx, lat_range_idx].to_numpy()[::-1, :]
        )
        # we have flipped the vertical directions as in ARTS 2.6 the pressure
        # must be ordered from high to low

    elif temp.ndim == 2:
        rawdata[varname] = np.float64(temp[central_idx, lat_range_idx].to_numpy())
    else:
        raise ValueError("There should be only 2d and 3d data...mmmh")


# %% get atms on the same altitude grid

N_profiles = len(rawdata["latitude"])

z_default = np.mean(rawdata["z"], axis=1)
z_default = z_default[z_default < alt_max]

z_org = rawdata["z"] * 1.0

for key in rawdata:

    if rawdata[key].ndim == 2:

        if np.size(rawdata[key], axis=0) >= len(z_default):

            print(f"reinterpolating {key}")

            data = np.zeros((len(z_default), N_profiles))
            for i in range(N_profiles):
                F_int = interp1d(
                    z_org[:, i], rawdata[key][:, i], fill_value="extrapolate"
                )
                data[:, i] = F_int(z_default)

            rawdata[key] = data


# %% now prepare data for arts
# This means we have to convert relative humidity to vmr and
# create batch_atm_compactand aux data


# get all 2d variables that is not atm_fieldnames
aux2d_fieldnames = [
    var
    for var in rawdata.keys()
    if var not in atm_fieldnames
    and np.size(rawdata[var], axis=0) == np.size(rawdata["p_grid"], axis=0)
]

# get all 1d variables that is not atm_fieldnames
aux1d_fieldnames = [
    var
    for var in rawdata.keys()
    if var not in atm_fieldnames and rawdata[var].ndim == 1
]

batch_atms = pa.arts.ArrayOfGriddedField4()
batch_aux2d = pa.arts.ArrayOfGriddedField2()
batch_aux1d = pa.arts.ArrayOfGriddedField1()

# allocate atm object
atm = pa.arts.GriddedField4()
atm.set_grid(0, atm_fieldnames)
atm.set_grid(1, rawdata["p_grid"][:, 0])
atm.data = np.zeros((len(atm.grids[0]), len(atm.grids[1]), 1, 1))

# allocate aux2d object
aux2d = pa.arts.GriddedField2()
aux2d.set_grid(0, aux2d_fieldnames)
aux2d.set_grid(1, rawdata["p_grid"][:, 0])
aux2d.data = np.zeros((len(aux2d.grids[0]), len(aux2d.grids[1])))

# allocate aux1d object
aux1d = pa.arts.GriddedField1()
aux1d.set_grid(0, aux1d_fieldnames)
aux1d.data = np.zeros(len(aux1d.grids[0]))


for i in range(N_profiles):

    if i % 50 == 0:
        print(f"processing profile {i} of {N_profiles}")

    atm.set_grid(1, rawdata["p_grid"][:, i])
    atm.data = np.zeros((len(atm.grids[0]), len(atm.grids[1]), 1, 1))

    for j, var in enumerate(atm_fieldnames):
        if var in rawdata:
            atm.data[j, :, 0, 0] = rawdata[var][:, i]

        # Convert mass densities from g m^{-3} to kg m^{-3}
        if "mass_density" in var:
            atm.data[j, :, 0, 0] /= 1000

        # convert relative humidity to vmr
        if var == "abs_species-H2O":
            temp = (
                atm.data[j, :, 0, 0]
                * e_eq_mixed_mk(atm.data[0, :, 0, 0])
                / atm.grids[1][:]
            )
            atm.data[j, :, 0, 0] = temp

        if var == "abs_species-O2":
            atm.data[j, :, 0, 0] = O2vmr

        if var == "abs_species-N2":
            atm.data[j, :, 0, 0] = N2vmr


    batch_atms.append(deepcopy(atm))

    # now the aux data
    for j, var in enumerate(aux2d_fieldnames):
        aux2d.data[j, :] = rawdata[var][:, i]

    batch_aux2d.append(deepcopy(aux2d))

    for j, var in enumerate(aux1d_fieldnames):
        aux1d.data[j] = rawdata[var][i]

    batch_aux1d.append(deepcopy(aux1d))


# %% before we save the data, let's do some checks

# calculate water columns and plot them

columns = {}
col_names = ["LWP", "IWP", "RWP", "SWP", "GWP", "HWP", "IWV"]
content_names = ["LWC", "IWC", "RWC", "SWC", "GWC", "HWC", "H2O"]
for name in col_names:
    columns[name] = np.zeros(N_profiles)


for i in range(N_profiles):

    for j, name in enumerate(col_names):

        names = [str(name) for name in batch_atms[i].grids[0]]
        idx = [k for k in range(len(names)) if content_names[j] in names[k]]

        if len(idx) > 1:
            idx = [idx_k for idx_k in idx if "mass" in names[idx_k]][0]
        else:
            idx = idx[0]

        if name == "IWV":
            # WV=rawdata.vmr_h2o./T.*p/R_h2o;
            density = (
                batch_atms[i].data[idx, :, 0, 0]
                / batch_atms[i].data[0, :, 0, 0]
                * batch_atms[i].grids[1]
                / gas_constant_water_vapor
            )

            columns[name][i] = np.trapezoid(density, x=batch_atms[i].data[1, :, 0, 0])
        else:

            # breakpoint()

            columns[name][i] = np.trapezoid(
                batch_atms[i].data[idx, :, 0, 0],
                x=batch_atms[i].data[1, :, 0, 0],
            )


# %% plot vertical coluns for check for checks

lat = np.array([a.data[0] for a in batch_aux1d])


# plt.style.use('ggplot')
fig, ax = plt.subplots(4, 2, figsize=(16, 10))


for i, name in enumerate(col_names):
    ax_k = ax[i // 2, i % 2]
    ax_k.plot(lat, columns[name], label=name)
    ax_k.set_title(name)
    ax_k.set_xlabel("latitude / °")
    ax_k.set_ylabel("column / kg m$^{-1}$ ")


# %% Create 'dropsonde data'
# simply select one of these profiles
# to keep it simple we simply take the middle

idx_selected = N_profiles // 2

dropsonde = pa.arts.GriddedField4()
dropsonde.set_grid(0, ["T", "z", "abs_species-H2O"])
dropsonde.set_grid(1, batch_atms[idx_selected].grids[1][:])
dropsonde.data = np.zeros((3, len(dropsonde.grids[1]), 1, 1))
dropsonde.data[0, :, 0, 0] = batch_atms[idx_selected].data[0, :, 0, 0]
dropsonde.data[1, :, 0, 0] = batch_atms[idx_selected].data[1, :, 0, 0]
dropsonde.data[2, :, 0, 0] = batch_atms[idx_selected].data[14, :, 0, 0]

# add some noise
rng = np.random.default_rng(12345)
T_noise_free = dropsonde.data[0, :, 0, 0] * 1.0
dropsonde.data[0, :, 0, 0] += rng.normal(0, 0.5, len(dropsonde.grids[1]))+T_offset

vmr_noise_free = dropsonde.data[2, :, 0, 0] * 1.0
temp = np.log10(dropsonde.data[2, :, 0, 0])
temp += rng.normal(0, 0.05, len(dropsonde.grids[1]))
dropsonde.data[2, :, 0, 0] = 10**temp


# plot dropsonde data
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))

ax2[0].plot(dropsonde.data[0, :, 0, 0], dropsonde.grids[1] / 1e3, label="obs")  # T
ax2[0].plot(T_noise_free, dropsonde.grids[1] / 1e3, label="true")
ax2[0].set_title("Temperature")
ax2[0].set_xlabel("T / K")
ax2[0].set_ylabel("p / hPa")
ax2[0].set_yscale("log")
ax2[0].invert_yaxis()
ax2[0].legend()

ax2[1].loglog(dropsonde.data[2, :, 0, 0], dropsonde.grids[1] / 1e3, label="obs")  # H2O
ax2[1].loglog(vmr_noise_free, dropsonde.grids[1] / 1e3, label="true")
ax2[1].set_title("H2O")
ax2[1].set_xlabel("H2O / vmr")
ax2[1].set_ylabel("p / hPa")
ax2[1].invert_yaxis()
ax2[1].legend()



# %% now plot profles for check

plot_vars = ["T", "abs_species-H2O"] + [
    name for name in atm_fieldnames if "mass_density" in name
]
# plot_vars=['T']

fig1, ax1 = plt.subplots(4, 2, figsize=(16, 10), sharey=True, sharex=True)

p_grid = np.mean([a.grids[1] for a in batch_atms], axis=0)


for i, name in enumerate(plot_vars):

    print(f"{name} - {i}")

    ax_k = ax1[i // 2, i % 2]

    names = [str(name) for name in batch_atms[0].grids[0]]
    idx = [k for k in range(len(names)) if name in names[k]][0]

    print(f"idx: {idx}")

    data = np.array([a.data[idx, :, 0, 0] for a in batch_atms])

    if name == "T":
        data -= np.mean(data, axis=0)
        # data -= dropsonde.data[0,:,0,0]
        cmap = "YlOrRd"
        pcm = ax_k.pcolormesh(
            lat,
            p_grid / 1e3,
            data.T,
            cmap=cmap,
            clim=[np.min(data), np.max(data)],
            rasterized=True,
        )

    else:
        data[data < 1e-10] = 1e-10
        data = np.log10(data)
        cmap = "Blues"
        pcm = ax_k.pcolormesh(
            lat, p_grid / 1e3, data.T, cmap=cmap, clim=[-6, 1], rasterized=True
        )

    ax_k.set_title(name)
    ax_k.set_xlabel("latitude / °")
    ax_k.set_ylabel("p / hPa")
    ax_k.set_yscale("log")

    cbar = fig1.colorbar(pcm, ax=ax_k)

    if name == "T":
        cbar.set_label("T - T$_{mean}$ / K")
    elif "mass_density" in name:
        cbar.set_label(" 10$^x$ kg m$^{-3}$")
    elif name == "abs_species-H2O":
        cbar.set_label("10$^x$ vmr")


ax_k.invert_yaxis()



# %% save data

batch_atms.savexml("atmosphere/atmospheres_true.xml")
batch_aux1d.savexml("atmosphere/aux1d_true.xml")
batch_aux2d.savexml("atmosphere/aux2d_true.xml")

dropsonde.savexml("observation/dropsonde.xml")

# %% save figures

fig1.savefig("check_plots/profiles.pdf")
fig.savefig("check_plots/columns.pdf")
fig2.savefig("check_plots/dropsonde.pdf")
