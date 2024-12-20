#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:37:19 2024

@author: u242031
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pyarts as pa

# %% path/constants

atm_data_folder= ('/home/zmaw/u242031/.cache/arts/arts-xml-data-2.6.8/'
                 +'planets/Earth/ECMWF/IFS/Eresmaa_137L')
batch_atm_compact_name='eresmaal137_all_q.xml.gz'
batch_atm_sfc_name='eresmaal137_all_q_surface.xml.gz'
batch_atm_aux_name='eresmaal137_all_q_extra.xml.gz'



f_grid=np.linspace(89,90,2)*1e9

# %% load some atm data

batch_atm=pa.xml.load(os.path.join(atm_data_folder,batch_atm_compact_name))
batch_sfc=pa.xml.load(os.path.join(atm_data_folder,batch_atm_sfc_name))
batch_aux=pa.xml.load(os.path.join(atm_data_folder,batch_atm_aux_name))




# %% create ARTS workspace

def setup_arts():

    ws=pa.workspace.Workspace()
    ws.verbositySetScreen(level=2)
    ws.verbositySetAgenda(level=0)



    # select/define agendas
    # =============================================================================

    ws.PlanetSet(option="Earth")

    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.water_p_eq_agendaSet()
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.ppath_agendaSet(option="FollowSensorLosPath")

    # define environment
    # =============================================================================

    #Stokes dim
    ws.stokes_dim=1

    # No jacobian calculations
    ws.jacobianOff()

    ws.cloudboxOff()


    # set absorption
    #==============================================================================
    ws.abs_speciesSet( species=[
        "H2O-PWR2022",
        "O3",
        "O2-PWR2022",
        "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        ])





    return ws

def setup_absorption(ws, f_grid, species=[]):

    if len(species)>0:
        ws.abs_speciesSet(species=species)

    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.propmat_clearsky_agendaAuto()


# %%

ws=setup_arts()

ws.f_grid=f_grid

setup_absorption(ws, f_grid)



