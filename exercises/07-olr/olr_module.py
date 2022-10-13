"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import numpy as np
import pyarts.workspace
from typhon import physics as phys


def calc_olr(atmfield, nstreams=10, fnum=300, fmin=1.0, fmax=75e12, verbosity=0):
    """Calculate the outgoing-longwave radiation for a given atmosphere.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray: Frequency grid [Hz], OLR [Wm^-2]
    """
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.LegacyContinuaInit()
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Definition of species
    ws.abs_speciesSet(
        species=[
            "H2O,H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
            "CO2, CO2-CKDMT252",
        ]
    )

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
       basename="lines/"
    )

    # Read cross section data
    ws.ReadXsecData(basename="lines/")


    # ws.abs_lines_per_speciesSetLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    
    
    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Calculate absorption
    ws.propmat_clearsky_agendaAuto()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)
    
    # Atmosphere and surface
    ws.atm_fields_compact = atmfield
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # Set surface height and temperature equal to the lowest atmosphere level
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]

    # Output radiance not converted
    ws.StringSet(ws.iy_unit, "1")

    # set cloudbox to full atmosphere
    ws.cloudboxSetFullAtm()

    # set particle scattering to zero, because we want only clear sky
    ws.scat_data_checked = 1
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()


    # No sensor properties
    ws.sensorOff()

    # No jacobian calculations
    ws.jacobianOff()
   
    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    # Perform RT calculations
    ws.DisortCalcIrradiance(nstreams=nstreams, emission=1)

    olr = ws.spectral_irradiance_field.value[:, -1, 0, 0, 1][:]

    return ws.f_grid.value[:], olr



def Change_T_with_RH_const(atmfield,DeltaT=0.):
    """Change the temperature everywhere in the atmosphere by a value of DeltaT
       but without changing the relative humidity. This results in a changed
       volume mixing ratio of water vapor.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        DeltaT (float): Temperature change [K].

    Returns:
        GriddedField4: Atmosphere field
    """

    #water vapor
    vmr = atmfield.get("abs_species-H2O")

    #Temperature
    T = atmfield.get("T")

    #Reshape pressure p, so that p has the same dimensions
    p = atmfield.grids[1][:].reshape(T.shape)

    #Calculate relative humidity
    rh = phys.vmr2relative_humidity(vmr, p, T)

    #Calculate water vapor volume mixing ratio for changed temperature
    vmr = phys.relative_humidity2vmr(rh, p, T+DeltaT)

    #update atmosphere field
    atmfield.set("T", T+DeltaT)
    atmfield.set("abs_species-H2O", vmr)

    return atmfield

