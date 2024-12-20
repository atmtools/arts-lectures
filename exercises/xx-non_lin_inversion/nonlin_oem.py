"""Perform an OEM retrieval and plot the results."""

from copy import deepcopy
import numpy as np
import pyarts.workspace
import pyarts as pa
from pyarts import xml


def basic_setup(f_grid, version="2.6.8", verbosity=0):
    """Create a basic ARTS workspace for radiative transfer calculations.

    Parameters:
        f_grid (ndarray): Frequency grid [Hz].
        version (str): ARTS version.
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        pyarts.workspace.Workspace: ARTS workspace.
    """

    pyarts.cat.download.retrieve(verbose=True, version=version)

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # standard emission agenda
    ws.iy_main_agendaSet(option="Emission")

    # cosmic background radiation
    ws.iy_space_agendaSet(option="CosmicBackground")

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    # sensor-only path
    ws.ppath_agendaSet(option="FollowSensorLosPath")

    # no refraction
    ws.ppath_step_agendaSet(option="GeometricPath")

    # Non reflecting surface
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    #########################################################################

    # Definition of absorption species
    ws.abs_speciesSet(
        species=[
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "O2-TRE05",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        ]
    )

    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesNormalization(option=normalization)

    # Set the frequency grid
    ws.f_grid = f_grid

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # on-the-fly absorption
    ws.propmat_clearsky_agendaAuto()

    # No sensor properties
    ws.sensorOff()

    # no jacobian
    ws.jacobianOff()

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    ws.AtmosphereSet1D()

    return ws


def forward_model(ws, atm_fields_compact, surface_reflectivity, sensor_pos, sensor_los):
    """Perform a forward model calculation.

    Parameters:
        ws (pyarts.workspace.Workspace): ARTS workspace.
        atm_fields_compact (pyarts.workspace.Field): ARTS field object.
        surface_reflectivity (float): Surface reflectivity.
        sensor_pos (ndarray): Sensor position [m].
        sensor_los (ndarray): Sensor line of sight [deg].

    Returns:
        ndarray: Brightness temperatures.
    """

    #########################################################################

    # Atmosphere and surface

    ws.atm_fields_compact = atm_fields_compact
    ws.atm_fields_compactAddConstant(
        ws.atm_fields_compact, "abs_species-N2", 0.78, 0, ["abs_species-H2O"]
    )
    ws.atm_fields_compactAddConstant(
        ws.atm_fields_compact, "abs_species-O2", 0.21, 0, ["abs_species-H2O"]
    )
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    #########################################################################

    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, surface_reflectivity)

    # Definition of sensor position and line of sight (LOS)
    ws.MatrixSet(ws.sensor_pos, sensor_pos)
    ws.MatrixSet(ws.sensor_los, sensor_los)

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Perform RT calculations
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.yCalc()

    return ws.y.value[:].copy()


# %% aux functions


def create_pertuberation(atm, variables, pertuberations):
    """Create a pertuberation of the atmospheric state.

    Parameters:
        atm (pyarts.workspace.Field): ARTS field object.
        variables (list): List of variables to pertubate.
        pertuberations (list): List of pertuberations.

    Returns:
        pyarts.workspace.Field: Pertubated ARTS field object.
    """

    # check if variables and pertuberations have the same length
    if len(variables) != len(pertuberations):
        raise ValueError("variables and pertuberations must have the same length")

    # check if pertuberation is either a scalar or a 1D array of the same length the pressure grid in atm
    for p in pertuberations:
        if not (np.isscalar(p) or (len(p) == atm.get_grid("p"))):
            raise ValueError(
                "pertuberations must be either a scalar or a 1D array\n of the same length the pressure grid in atm"
            )

    # create a copy of the atm
    atm_perturbed = deepcopy(atm)

    # add pertuberation to the variables
    for v, p in zip(variables, pertuberations):

        perp = atm.get(v, keep_dims=False) + p
        atm_perturbed.set(v, np.array(perp[np.newaxis, :, np.newaxis, np.newaxis]))

    return atm_perturbed


def create_apriori_covariance_matrix(x, z, delta_x, correlation_length):
    """Create the a priori covariance matrix.

    Parameters:
        x (ndarray): A priori state.
        z (ndarray): Altitude grid [m].
        delta_x (float or ndarray): Variance of the state.
        correlation_length (float or ndarray): Correlation length.

    Returns:
        ndarray: A priori covariance matrix.
    """    

    # check if x and z have the same length
    if len(x) != len(z):
        raise ValueError("x and z must have the same length")

    # check if correlation_length is either a scalar or a 1D array of the same length the pressure grid in atm
    if not (np.isscalar(correlation_length) or (len(correlation_length) == len(z))):
        raise ValueError(
            "Correlation length must be either a scalar or a 1D array\n of the same length as z"
        )

    # check if delta_x is either a scalar or a 1D array of the same length the pressure grid in atm
    if not (np.isscalar(delta_x) or (len(delta_x) == len(z))):
        raise ValueError(
            "variance_ii must be either a scalar or a 1D array\n of the same length as z"
        )

    # create a covariance matrix
    S_x = np.zeros((len(z), len(z)))

    # fill the covariance matrix
    for i in range(len(z)):

        if np.isscalar(correlation_length):
            S_x[i, :] = delta_x * np.exp(-np.abs(z - z[i]) / correlation_length)
        else:
            S_x[i, :] = delta_x * np.exp(-np.abs(z - z[i]) / correlation_length[i])

    return S_x


# %%

# %%
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # define sensor
    fmin = 116e9
    fmax = 121e9
    fnum = 41
    f_grid = np.linspace(fmin, fmax, fnum)

    sensor_pos = np.array([[800e3]])
    sensor_los = np.array([[180.0]])

    NeDT = 0.5  # K

    # load eresmaa dataset
    atms = pa.arts.ArrayOfGriddedField4()
    atms.readxml("planets/Earth/ECMWF/IFS/Eresmaa_137L/eresmaal137_all_t.xml.gz")

    sfcs = pa.arts.Matrix()
    sfcs.readxml(
        "planets/Earth/ECMWF/IFS/Eresmaa_137L/eresmaal137_all_t_surface.xml.gz"
    )

    # select one profile
    idx = 1234

    # this will be the a priori state
    atm = atms[idx]
    sfc = sfcs[idx, :]

    # create a pertuberation, this ww want to retrieve
    atm_perturbed = create_pertuberation(atm, ["T"], [10.0])

    surface_reflectivity = 0.4

    ws = basic_setup(f_grid)
    y_apr = forward_model(ws, atm, surface_reflectivity, sensor_pos, sensor_los)
    y = forward_model(ws, atm_perturbed, surface_reflectivity, sensor_pos, sensor_los)

    # %% set covariance matrices

    # measurement covariance
    S_y = np.diag(np.pnes(len(f_grid)) * NeDT)
