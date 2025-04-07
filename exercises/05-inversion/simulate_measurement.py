#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:11:05 2024

@author: Manfred Brath
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import pyarts as pa
import oem



# %% paths/constants

# Load the a priori atmospheric state.
atm_fields = pa.xml.load("input/x_apriori.xml")
z = atm_fields.get("z", keep_dims=False)
x_apriori = atm_fields.get("abs_species-H2O", keep_dims=False)

nedt=0.05 #K
#%% prepare measurement simulation

#define channels
f_min=22.15e9
f_max=22.30e9
f_num=2000
f_grid=np.linspace(f_min,f_max,f_num)


#define "true" atmosphere
atm_true=deepcopy(atm_fields)

#add pertuberation to the water vapor profile
vmr_h2o=atm_true.get("abs_species-H2O", keep_dims=False)
vmr_h2o_perp=vmr_h2o+3e-5*np.exp(-z/50000)*0.5*np.sin(2*np.pi*0.0001*z)+0.5e-5
vmr_h2o_perp[vmr_h2o_perp<0]=vmr_h2o[-1]

atm_true.set("abs_species-H2O", np.array(vmr_h2o_perp[np.newaxis,:,np.newaxis,np.newaxis]))

plt.style.use('seaborn-v0_8')

fig, ax =plt.subplots(1,2)

ax[0].plot(atm_fields.get("abs_species-H2O", keep_dims=False), z/1000, label='a priori')
ax[0].plot(atm_true.get("abs_species-H2O", keep_dims=False), z/1000, label='perturbed')
ax[0].set_ylabel('altitude / km')
ax[0].set_xlabel(r'$vmr_{\text{H}_2 \text{O} }$')
ax[0].legend()

ax[1].plot(atm_true.get("T", keep_dims=False), z/1000)
ax[1].set_xlabel('temperature / K')

fig.tight_layout()


# %% simulate measurement

y_apr, K_apr = oem.forward_model(f_grid, atm_fields)
y, K = oem.forward_model(f_grid, atm_true)

# %%add noise
#we use a seed to be deterministic
rng = np.random.default_rng(12345)
y_obs = y+rng.normal(scale=nedt,size=f_num)

fig, ax =plt.subplots(1,1)
ax.plot(f_grid,y_apr,label='a priori')
ax.plot(f_grid,y,label='simulation')
ax.plot(f_grid,y_obs,label='measurement')

#save measurement
measurement=pa.arts.GriddedField1()
measurement.grids=[f_grid]
measurement.gridnames[0]='Frequency'
measurement.data=y_obs
measurement.savexml("input/measurement.xml")

#save "true profile"
atm_true.savexml("input/x_true.xml")

