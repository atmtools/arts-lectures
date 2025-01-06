"""Perform an OEM retrieval and plot the results."""

from copy import deepcopy
import numpy as np
import pyarts.workspace
import pyarts as pa
from pyarts import xml


def basic_setup(f_grid, version="2.6.8", verbosity=0):
    """Set up a basic ARTS workspace configuration for radiative transfer calculations.
    This function initializes an ARTS workspace with standard settings for atmospheric
    radiative transfer calculations, particularly focused on emission-based calculations
    in Earth's atmosphere.
    Parameters
    ----------
    f_grid : numpy.ndarray
        Frequency grid for calculations [Hz]
    version : str, optional
        Version of ARTS catalog to use (default is "2.6.8")
    verbosity : int, optional
        Level of output verbosity (default is 0)
    Returns
    -------
    pyarts.workspace.Workspace
        Configured ARTS workspace with:
        - Earth as planet
        - Emission-based radiative transfer
        - Cosmic background radiation
        - No refraction
        - Non-reflecting surface
        - Single Stokes component
        - Basic absorption species (H2O, O2, N2)
        - Planck brightness temperature as output unit
        - Clear-sky conditions
    Notes
    -----
    The function performs the following main steps:
    1. Set up emission-based calculations
    2. Include cosmic background radiation
    3. Use Earth as planet
    4. Set up non-reflecting surface
    5. Set up single Stokes component
    6. Define basic absorption species
    7. Set up Planck brightness temperature as output unit
    8. Set up clear-sky conditions
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
    # We use predefined models for H2O, O2, and N2 from Rosenkranz
    # as they are fast and accurate for microwave and tropospheric retrieval.
    ws.abs_speciesSet(
        species=[
            "H2O-PWR2022",
            "O2-PWR2022",
            "N2-SelfContPWR2021",
        ]
    )

    ws.abs_lines_per_speciesSetEmpty()

    # Set the frequency grid
    ws.f_grid = f_grid

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


def forward_model(
    ws,
    atm_fields_compact,
    surface_reflectivity,
    surface_temperature,
    sensor_pos,
    sensor_los,
    retrieval_quantity="",
):
    """
    Performs radiative transfer calculations using ARTS (Atmospheric Radiative Transfer Simulator).
    This function sets up and executes forward model calculations, optionally including Jacobian
    calculations for retrievals of water vapor (H2O) or temperature (T).
    Parameters
    ----------
    ws : arts.Workspace
        ARTS workspace object containing the computational environment.
    atm_fields_compact : arts.AtmFieldsCompact
        Compact representation of atmospheric fields.
    surface_reflectivity : float
        Surface reflectivity value (between 0 and 1).
    surface_temperature : float
        Surface temperature value in Kelvin.
    sensor_pos : list or numpy.ndarray
        Sensor position coordinates.
    sensor_los : list or numpy.ndarray
        Sensor line of sight angles.
    retrieval_quantity : str, optional
        Specifies the quantity to be retrieved. Must be either 'H2O' or 'T'.
        If empty, no Jacobian is calculated.
    Returns
    -------
    tuple
        - numpy.ndarray: Calculated radiances (ws.y value)
        - arts.Matrix or empty Matrix: Jacobian matrix if retrieval_quantity is specified,
          otherwise an empty matrix
    Notes
    -----
    The function performs the following main steps:
    1. Sets up atmospheric fields including N2 and O2 if not present
    2. Configures surface properties
    3. Sets sensor position and line of sight
    4. Calculates Jacobians if retrieval_quantity is specified
    5. Performs radiative transfer calculations in clear-sky conditions
    The calculation assumes no scattering (cloudbox is turned off).
    Raises
    ------
    ValueError
        If retrieval_quantity is neither 'H2O' nor 'T' when specified.
    """

    #########################################################################

    # Atmosphere and surface

    ws.atm_fields_compact = atm_fields_compact

    # check if N2 and O2 in atm_fields_compact
    if not "abs_species-N2" in atm_fields_compact.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-N2", 0.78, 0, ["abs_species-H2O"]
        )

    if not "abs_species-O2" in atm_fields_compact.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-O2", 0.21, 0, ["abs_species-H2O"]
        )

    # ws.atm_fields_compactAddConstant(
    #     ws.atm_fields_compact, "abs_species-N2", 0.78, 0, ["abs_species-H2O"]
    # )
    # ws.atm_fields_compactAddConstant(
    #     ws.atm_fields_compact, "abs_species-O2", 0.21, 0, ["abs_species-H2O"]
    # )
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    ws.Extract(ws.z_surface, ws.z_field, 0)
    # ws.Extract(ws.t_surface, ws.t_field, 0)
    ws.t_surface = [[surface_temperature]]

    #########################################################################

    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, surface_reflectivity)

    # Definition of sensor position and line of sight (LOS)
    ws.MatrixSet(ws.sensor_pos, sensor_pos)
    ws.MatrixSet(ws.sensor_los, sensor_los)

    # Jacobian calculation
    if len(retrieval_quantity) > 0:
        ws.jacobianInit()
        if retrieval_quantity == "H2O":
            ws.jacobianAddAbsSpecies(
                g1=ws.p_grid,
                g2=ws.lat_grid,
                g3=ws.lon_grid,
                species="H2O-PWR2022",
                unit="vmr",
            )
        elif retrieval_quantity == "T":
            ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
        else:
            raise ValueError("only H2O or T are allowed as retrieval quantity")
        ws.jacobianClose()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Perform RT calculations
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.yCalc()

    if len(retrieval_quantity) > 0:
        jacobian = ws.jacobian.value[:].copy()
    else:
        jacobian = pa.arts.Matrix()

    return ws.y.value[:].copy(), jacobian


def Forward_model(
    f_grid,
    atm_fields_compact,
    surface_reflectivity,
    surface_temperature,
    sensor_pos,
    sensor_los,
    retrieval_quantity="",
):
    """
    Computes the forward model and its Jacobian for atmospheric radiative transfer.
    This function sets up and runs a radiative transfer calculation using the ARTS model,
    computing both simulated measurements and their derivatives with respect to the
    retrieval parameters.
    Parameters
    ----------
    f_grid : array-like
        Frequency grid for the radiative transfer calculation [Hz].
    atm_fields_compact : dict
        Dictionary containing atmospheric fields (temperature, humidity, etc.).
    surface_reflectivity : float or array-like
        Surface reflectivity value(s).
    surface_temperature : float
        Surface temperature [K].
    sensor_pos : array-like
        Sensor position coordinates [m].
    sensor_los : array-like
        Sensor line-of-sight angles [degrees].
    retrieval_quantity : str, optional
        Specifies the quantity to be retrieved. Must be either 'H2O' or 'T'.
        If empty, no Jacobian is calculated.
    Returns
    -------
    tuple
        - y : array-like
            Simulated measurements (e.g., brightness temperatures).
        - jacobian : array-like
            Jacobian matrix containing the derivatives of measurements
            with respect to retrieval parameters.
    Notes
    -----
    This function serves as a wrapper around the ARTS radiative transfer model,
    providing a simplified interface for forward model calculations.
    """

    ws = basic_setup(f_grid)

    y, jacobian = forward_model(
        ws,
        atm_fields_compact,
        surface_reflectivity,
        surface_temperature,
        sensor_pos,
        sensor_los,
        retrieval_quantity=retrieval_quantity,
    )

    return y, jacobian


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
    """Create an a priori covariance matrix using an exponential correlation function.
    This function creates a covariance matrix for a given atmospheric profile using an
    exponential correlation function. The matrix elements are calculated based on the
    vertical distance between levels and a specified correlation length.
    Parameters
    ----------
    x : array-like
        The state vector (e.g., atmospheric parameter values).
    z : array-like
        The altitude levels corresponding to the state vector.
    delta_x : float or array-like
        The variance (diagonal elements) of the covariance matrix. Can be either
        a scalar applied to all levels or a 1D array with values for each level.
    correlation_length : float or array-like
        The correlation length determining how quickly correlations decay with
        distance. Can be either a scalar applied to all levels or a 1D array
        with values for each level.
    Returns
    -------
    numpy.ndarray
        The a priori covariance matrix with shape (len(z), len(z)).
    Raises
    ------
    ValueError
        If x and z have different lengths, or if delta_x or correlation_length
        arrays don't match the length of z when provided as arrays.
    Notes
    -----
    The covariance matrix is constructed using the exponential correlation function:
    S_ij = delta_x * exp(-|z_i - z_j|/correlation_length)
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

        if np.isscalar(correlation_length) and np.isscalar(delta_x):
            S_x[i, :] = delta_x * np.exp(-np.abs(z - z[i]) / correlation_length)
        elif np.isscalar(correlation_length):
            S_x[i, :] = delta_x[i] * np.exp(-np.abs(z - z[i]) / correlation_length)
        elif np.isscalar(delta_x):
            S_x[i, :] = delta_x * np.exp(-np.abs(z - z[i]) / correlation_length[i])
        else:
            S_x[i, :] = delta_x[i] * np.exp(-np.abs(z - z[i]) / correlation_length[i])

    return S_x


def set_correlation_length(z, len_sfc, len_toa=None):
    """
    Calculate the correlation length for each altitude level.
    This function computes a correlation length profile that can either be constant
    or vary linearly with altitude. The correlation length can be specified at the
    surface and optionally at the top of the atmosphere.
    Parameters
    ----------
    z : numpy.ndarray
        Altitude levels [m]
    len_sfc : float
        Correlation length at the surface [m]
    len_toa : float, optional
        Correlation length at the top of atmosphere [m]. If None, a constant
        correlation length equal to len_sfc is used (default: None)
    Returns
    -------
    numpy.ndarray
        Correlation length for each altitude level [m]
    Notes
    -----
    If len_toa is provided, the correlation length varies linearly between len_sfc
    at the surface and len_toa at the top of the atmosphere. If len_toa is None,
    a constant correlation length equal to len_sfc is used for all levels.
    """

    if len_toa is None:
        correlation_length = np.ones_like(z) * len_sfc
    else:
        correlation_length = (len_toa - len_sfc) / (z[-1] - z[0]) * z + len_sfc

    return correlation_length


# %%


def Hamp_channels_simplified(Band):
    """
    Get simplified HAMP radiometer channel frequencies and noise specifications.
    This function provides a simplified set of frequency channels and noise
    characteristics for different bands of the HAMP (HALO Microwave Package)
    radiometer.
    Parameters
    ----------
    Band : str
        The radiometer band to get specifications for.
        Valid options are:
        - "H2O": Water vapor band
        - "O2_low": Lower oxygen band
        - "O2_high": Higher oxygen band
    Returns
    -------
    tuple
        A tuple containing:
        - f_grid : numpy.ndarray
            Frequency channels in Hz
        - NeDT : float
            Noise equivalent differential temperature in Kelvin
    Raises
    ------
    ValueError
        If the specified Band is not one of the valid options.
    Notes
    -----
    The frequencies are returned in Hz and represent simplified channel selections
    for each band of the HAMP radiometer.
    """

    if Band == "H2O":
        f_grid = np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]) * 1e9  # Hz
        NeDT = 0.1  # K
    elif Band == "O2_low":
        f_grid = np.array([50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]) * 1e9
        NeDT = 0.2  # K
    elif Band == "O2_high":
        f_grid = (118.75 + np.array([1.3, 2.3, 4.2, 8.5])) * 1e9
        NeDT = 0.6
    else:
        raise ValueError("Band not defined")

    return f_grid, NeDT


# %%
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # surface temperature
    surface_temperature = 300  # K

    # surface reflectivity
    surface_reflectivity = 0.4

    # sensor position and line of sight
    sensor_pos = np.array([[15e3]])
    sensor_los = np.array([[180.0]])

    # load dropsonde data
    dropsonde = xml.load("observation/dropsonde.xml")

    # load frequency data for 50 GHz channels
    f_grid = xml.load("observation/f_grid_50GHz.xml")[:]

    y, jacobian = Forward_model(
        f_grid,
        dropsonde,
        surface_reflectivity,
        surface_temperature,
        sensor_pos,
        sensor_los,
        retrieval_quantity="T",
    )

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(f_grid / 1e9, y, "x")
    ax[0].set_ylabel("Brightness temperature [K]")
    ax[0].set_xlabel("Frequency [GHz]")

    ax[1].plot(jacobian.T, dropsonde.data[1, :, 0, 0] / 1e3)
    ax[1].set_ylabel("Jacobian")
    ax[1].set_xlabel("Altitude [km]")
    ax[1].set_title("Temperature jacbian")

    # load observation data
    y_obs = xml.load("observation/y_obs_50GHz.xml")[:]
