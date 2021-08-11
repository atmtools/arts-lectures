"""Calculate and plot zenith opacity and brightness temperatures. """
import re

import matplotlib.pyplot as plt
import numpy as np
import typhon as ty
import pyarts.workspace


def tags2tex(tags):
    """Replace all numbers in every species tag with LaTeX subscripts."""
    return [re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag) for tag in tags]


def run_arts(
    species,
    zenith_angle=0.0,
    height=0.0,
    fmin=10e9,
    fmax=250e9,
    fnum=1_000,
    verbosity=0,
):
    """Perform a radiative transfer simulation.

    Parameters:
        species (list[str]): List of species tags.
        zenith_angle (float): Viewing angle [deg].
        height (float): Sensor height [m].
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.

    Returns:
        ndarray, ndarray, ndarray:
          Frequency grid [Hz], Brightness temperature [K], Optical depth [1]
    """
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Agenda for scalar gas absorption calculation
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    # (standard) emission calculation
    ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)

    # cosmic background radiation
    ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)

    # on-the-fly absorption
    ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__OnTheFly)

    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)

    # no refraction
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # A pressure grid rougly matching 0 to 80 km, in steps of 2 km.
    ws.VectorNLogSpace(ws.p_grid, 100, 1013e2, 10.0)

    ws.abs_speciesSet(species=species)

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
       basename="spectroscopy/Artscat/"
    )

    # ws.abs_lines_per_speciesSetLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesSetCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesSetNormalization(option=normalization)
    
    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric scenario
    ws.AtmRawRead(basename="planets/Earth/Fascod/midlatitude-summer/midlatitude-summer")

    # Non reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.1)
    ws.Copy(
        ws.surface_rtprop_agenda,
        ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface,
    )

    # No sensor properties
    ws.sensorOff()

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    # Extract optical depth as auxiliary variables
    ws.ArrayOfStringSet(ws.iy_aux_vars, ["Optical depth"])

    # Atmosphere and surface
    ws.AtmosphereSet1D()
    ws.AtmFieldsCalc()
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    # Definition of sensor position and line of sight (LOS)
    ws.MatrixSet(ws.sensor_pos, np.array([[height]]))
    ws.MatrixSet(ws.sensor_los, np.array([[zenith_angle]]))

    # Perform RT calculations
    ws.abs_xsec_agenda_checkedCalc()
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.yCalc()

    return (ws.f_grid.value.copy(), ws.y.value.copy(), ws.y_aux.value[0].data.copy())
