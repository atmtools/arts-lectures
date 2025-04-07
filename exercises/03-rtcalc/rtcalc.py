# %% Import modules and define functions
"""Calculate and plot zenith opacity and brightness temperatures. """
import re
import numpy as np
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
    verbosity=0
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
    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    
    pyarts.cat.download.retrieve(verbose=bool(verbosity))
    
    
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.iy_main_agendaSet(option="Emission")
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    ws.verbositySetScreen(ws.verbosity, verbosity)

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
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric scenario
    ws.AtmRawRead(basename="planets/Earth/Fascod/midlatitude-summer/midlatitude-summer")

    # Non reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.1)
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

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
    ws.propmat_clearsky_agendaAuto()
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.yCalc()

    return (
        ws.f_grid.value[:].copy(),
        ws.y.value[:].copy(),
        ws.y_aux.value[0][:].copy(),
    )


# %% Run module as script
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # pyarts.cat.download.retrieve(verbose=True)

    species = ["N2", "O2", "H2O"]
    height = 0.0  # m
    zenith_angle = 0.0  # deg

    # Run the radiative transfer simulation to get the frequency grid,
    # brightness temperature and optical depth
    freq, bt, od = run_arts(species, zenith_angle, height)

    # Plot the zenith opacity with logarithmic scale on y axis
    fig, ax = plt.subplots()
    ax.semilogy(freq, od)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Zenith opacity")

    # Plot the brightness temperature
    fig, ax = plt.subplots()
    ax.plot(freq, bt)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Brightness temperature [K]")

    plt.show()
