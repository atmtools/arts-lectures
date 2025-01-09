#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARTS (Atmospheric Radiative Transfer Simulator) interface module for non-linear
temperature profile retrievals using the Optimal Estimation Method (OEM).

This module provides functions for:
1. Setting up and running ARTS radiative transfer calculations
2. Performing temperature retrievals using OEM
3. Handling covariance matrices and correlation lengths
4. Managing HAMP radiometer channel specifications

The module is designed for atmospheric temperature profile retrievals from
radiometric measurements, particularly focusing on airborne observations
like those from the HALO Microwave Package (HAMP).

Main Components:
--------------
ARTS Functions:
    - basic_setup: Configures ARTS workspace for radiative transfer
    - forward_model: Performs radiative transfer calculations
    - Forward_model: High-level wrapper for forward model calculations
    - set_sensor: Sets sensor position and line of sight
    - prepare_initial_conditions: Sets up initial atmospheric conditions
    - retrieval_T: Implements OEM retrieval for temperature
    - temperature_retrieval: High-level wrapper for temperature retrieval

Auxiliary Functions:
    - create_apriori_covariance_matrix: Generates covariance matrices
    - set_correlation_length: Calculates correlation length profiles
    - Hamp_channels_simplified: Provides HAMP radiometer specifications

Dependencies:
------------
- numpy: Numerical computations
- pyarts: ARTS radiative transfer model interface
- pyarts.workspace: ARTS computational environment
- pyarts.xml: XML file handling for ARTS

Notes:
The module assumes:
- Clear-sky conditions (no scattering)
- Earth as the planetary body
- Surface properties (temperature and reflectivity) are known
- Measurement noise characteristics are known

The retrieval is performed using the Levenberg-Marquardt algorithm
implemented in ARTS, with options for diagnostic output and convergence
monitoring.

created: Tue Jan 7 12:13:59 2025
@author: Manfred Brath
"""


from copy import deepcopy
import numpy as np
import pyarts.workspace
import pyarts as pa
from pyarts import xml

# %% ARTS functions

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
            ws.atm_fields_compact, "abs_species-N2", 0.7808, 0  # , ["abs_species-H2O"]
        )

    if not "abs_species-O2" in atm_fields_compact.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-O2", 0.2095, 0  # , ["abs_species-H2O"]
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


def set_sensor(ws, sensor_pos, sensor_los):
    ws.MatrixSet(ws.sensor_pos, np.array([[sensor_pos]]))
    ws.MatrixSet(ws.sensor_los, np.array([[sensor_los]]))


def prepare_initial_conditions(
    ws,
    atmosphere_apr,
    surface_temperature,
    surface_reflectivity,
    O2vmr=0.2095,
    N2vmr=0.7808,
):

    ws.atm_fields_compact = atmosphere_apr
    # check if N2 and O2 in background atmosphere
    if not "abs_species-N2" in atmosphere_apr.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-N2", N2vmr, 0, ["abs_species-H2O"]
        )

    if not "abs_species-O2" in atmosphere_apr.grids[0]:
        ws.atm_fields_compactAddConstant(
            ws.atm_fields_compact, "abs_species-O2", O2vmr, 0, ["abs_species-H2O"]
        )

    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    ws.t_surface = [[surface_temperature]]

    ws.Extract(ws.z_surface, ws.z_field, 0)
    try:
        ws.surface_scalar_reflectivity = surface_reflectivity
    except:
        ws.surface_scalar_reflectivity = [surface_reflectivity]


def retrieval_T(
    ws, y, S_y, S_a, max_iter=50, stop_dx=0.01, Diagnostics=False, Verbosity=False
):
    """
    Performs temperature retrieval using Optimal Estimation Method (OEM).
    This function performs a temperature retrieval using the Optimal Estimation Method
    with Levenberg-Marquardt optimization. It uses ARTS (Atmospheric Radiative Transfer
    Simulator) workspace for the radiative transfer calculations.
    Parameters
    ----------
    ws : pyarts.workspace.Workspace
        ARTS workspace object containing the atmospheric setup
    y : numpy.ndarray
        Measurement vector
    S_y : numpy.ndarray
        Measurement error covariance matrix
    S_a : numpy.ndarray
        A priori covariance matrix
    max_iter : int, optional
        Maximum number of iterations (default: 50)
    stop_dx : float, optional
        Convergence criterion for state vector update (default: 0.01)
    Diagnostics : bool, optional
        If True, returns additional diagnostic quantities (default: False)
    Verbosity : bool, optional
        If True, prints detailed output during retrieval (default: False)
    Returns
    -------
    dict
        Dictionary containing retrieval results:
        - 'x': Retrieved state vector
        - 'y_fit': Forward model output for retrieved state
        - 'S_o': Observation error covariance matrix
        - 'S_s': Smoothing error covariance matrix
        - 'dx_o': Observation error vector
        - 'dx_s': Smoothing error vector
        If Diagnostics=True, also includes:
        - 'A': Averaging kernel matrix
        - 'G': Gain matrix
    Notes
    -----
    The function assumes that the ARTS workspace is properly initialized with all
    necessary atmospheric and sensor parameters before the retrieval starts.
    """    

    # Copy the measeurement vector to the ARTS workspace
    ws.y = y

    # Switch off cloudbox only clear sky
    ws.cloudboxOff()

    # Start definition of retrieval quantities
    ###########################################################################
    ws.retrievalDefInit()

    # Add temperature as retrieval quantity#
    ws.retrievalAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)

    # Set a priori covariance matrix
    ws.covmat_sxAddBlock(block=S_a)

    # Set measurement error covariance matrix
    ws.covmat_seAddBlock(block=S_y)

    # Close retrieval definition
    ws.retrievalDefClose()
    ############################################################################

    # Initialise
    # x, jacobian and yf must be initialised
    ws.VectorSet(ws.x, [])
    ws.VectorSet(ws.yf, [])
    ws.MatrixSet(ws.jacobian, [])

    # Iteration agenda
    @pa.workspace.arts_agenda
    def inversion_iterate_agenda(ws):

        ws.Ignore(ws.inversion_iteration_counter)

        # Map x to ARTS' variables
        ws.x2artsAtmAndSurf()

        # To be safe, rerun some checks
        ws.atmfields_checkedCalc()
        ws.atmgeom_checkedCalc()

        # Calculate yf and Jacobian matching x.
        ws.yCalc(y=ws.yf)

    #
    ws.inversion_iterate_agenda = inversion_iterate_agenda

    # some basic checks
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    # create a priori
    ws.xaStandard()

    # Run OEM
    ws.OEM(
        method="lm",
        max_iter=max_iter,
        display_progress=int(Verbosity),
        stop_dx=stop_dx,
        lm_ga_settings=[100, 2, 3, 1e5, 1, 99],
    )
    #
    if Verbosity == True:
        ws.Print(ws.oem_errors, 0)

    oem_diagostics = ws.oem_diagnostics.value[:]
    if oem_diagostics[0] > 0:
        print(f"Convergence status:                    {oem_diagostics[0]}")
        print(f"Start value of cost function:          {oem_diagostics[1]}")
        print(f"End value of cost function:            {oem_diagostics[2]}")
        print(f"End value of y-part of cost function:  {oem_diagostics[3]}")
        print(f"Number of iterations:                  {oem_diagostics[4]}\n")

    # Compute averaging kernel matrix
    ws.avkCalc()

    # Compute smoothing error covariance matrix
    ws.covmat_ssCalc()

    # Compute observation system error covariance matrix
    ws.covmat_soCalc()

    # Extract observation errors
    ws.retrievalErrorsExtract()

    result = {}
    result["x"] = ws.x.value[:] * 1.0
    result["y_fit"] = ws.yf.value[:] * 1.0
    result["S_o"] = ws.covmat_so.value[:] * 1.0
    result["S_s"] = ws.covmat_ss.value[:] * 1.0
    result["dx_o"] = ws.retrieval_eo.value[:] * 1.0
    result["dx_s"] = ws.retrieval_ss.value[:] * 1.0
    if Diagnostics:
        result["A"] = ws.avk.value[:] * 1.0
        result["G"] = ws.dxdy.value[:] * 1.0

    return result


def temperature_retrieval(
    y_obs,
    f_grid,
    sensor_altitude,
    sensor_los,
    background_atmosphere,
    surface_temperature,
    surface_reflectivity,
    S_y,
    S_a,
    Diagnostics=False,
    Verbosity=False,
):
    """Retrieves atmospheric temperature profile using the Optimal Estimation Method (OEM).
    This function performs a non-linear retrieval of atmospheric temperature profiles from
    spectroscopic measurements using the OEM approach.
    Parameters
    ----------
    y_obs : array-like
        Observed spectral radiances
    f_grid : array-like
        Frequency grid points [Hz]
    sensor_altitude : array-like
        Position vector of the sensor [m]
    sensor_los : array-like
        Line of sight vector of the sensor
    background_atmosphere : dict
        Dictionary containing initial atmospheric state
    surface_temperature : float
        Surface temperature [K]
    surface_reflectivity : float
        Surface reflectivity [0-1]
    S_y : array-like
        Measurement error covariance matrix
    S_a : array-like
        A priori covariance matrix
    Diagnostics : bool, optional
        If True, returns additional diagnostic quantities (default is False)
    Verbosity : bool, optional
        If True, prints detailed progress information (default is False)
    Returns
    -------
    T_ret : array-like
        Retrieved temperature profile
    DeltaT : array-like
        Retrieval error (combination of observation and smoothing error)
    y_fit : array-like
        Fitted spectral radiances
    A : array-like, optional
        Averaging kernel matrix (only if Diagnostics=True)
    G : array-like, optional
        Gain matrix (only if Diagnostics=True)
    Notes
    -----
    The retrieval is performed using an iterative Levenberg-Marquardt algorithm
    implemented in the retrieval_T function.
    """

    ws = basic_setup(f_grid)

    prepare_initial_conditions(
        ws, background_atmosphere, surface_temperature, surface_reflectivity
    )

    set_sensor(ws, sensor_altitude, sensor_los)

    result = retrieval_T(
        ws, y_obs, S_y, S_a, Diagnostics=Diagnostics, Verbosity=Verbosity
    )

    T_ret = result["x"]
    DeltaT = np.sqrt(result["dx_o"] ** 2 + result["dx_s"] ** 2)
    y_fit = result["y_fit"]

    if Diagnostics:

        A = result["A"]
        G = result["G"]

        return T_ret, DeltaT, y_fit, A, G

    return T_ret, DeltaT, y_fit


# %% aux functions


def create_apriori_covariance_matrix(x, z, delta_x, correlation_length):
    """Creates a covariance matrix for a priori knowledge based on exponential correlation.
    This function generates a covariance matrix using exponential correlation between
    points, allowing for either constant or varying correlation lengths and standard
    deviations across the vertical grid.
    Parameters
    ----------
    x : array-like
        Horizontal coordinate grid points (must have same length as z)
    z : array-like 
        Vertical coordinate grid points
    delta_x : float or array-like
        Standard deviation(s) of the a priori knowledge. Can be either:
        - A scalar value applied to all points
        - A 1D array of length matching z with point-specific values
    correlation_length : float or array-like
        Correlation length(s) determining how quickly correlations decay with distance.
        Can be either:
        - A scalar value applied to all points
        - A 1D array of length matching z with point-specific values
    Returns
    -------
    numpy.ndarray
        The covariance matrix with shape (len(z), len(z))
    Raises
    ------
    ValueError
        If x and z have different lengths
        If correlation_length array length doesn't match z length
        If delta_x array length doesn't match z length
    Notes
    -----
    The covariance matrix is constructed using the exponential correlation function:
    S_ij = delta_x^2 * exp(-|z_i - z_j|/correlation_length)
    For varying delta_x and correlation_length, the values specific to each point
    are used in the calculation.
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
            S_x[i, :] = delta_x**2 * np.exp(-np.abs(z - z[i]) / correlation_length)
        elif np.isscalar(correlation_length):
            S_x[i, :] = delta_x[i] ** 2 * np.exp(-np.abs(z - z[i]) / correlation_length)
        elif np.isscalar(delta_x):
            S_x[i, :] = delta_x**2 * np.exp(-np.abs(z - z[i]) / correlation_length[i])
        else:
            S_x[i, :] = delta_x[i] ** 2 * np.exp(
                -np.abs(z - z[i]) / correlation_length[i]
            )

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

    # %% Test the forward model and some of the data

    import matplotlib.pyplot as plt

    # surface reflectivity
    # It is assumed that the surface reflectivity is known and constant
    surface_reflectivity = 0.4

    # surface temperature
    # It is assumed that the surface temperature is known and constant
    surface_temperature = 300.0  # K

    # define sensor positions and line of sight
    # we assume a HALO like airplane with a sensor at 15 km altitude and a line of sight of 180 degrees
    sensor_pos = np.array([[15e3]])
    sensor_los = np.array([[180.0]])

    # load dropsonde data
    dropsonde = xml.load("observation/dropsonde.xml")

    # load frequency data for 50 GHz channels
    f_grid = xml.load("observation/f_grid_50GHz.xml")[:]

    # ws = basic_setup(f_grid)
    # y, jacobian = forward_model(ws, dropsonde, surface_reflectivity, surface_temperature, sensor_pos, sensor_los,retrieval_quantity="T")

    # load observation data
    y_obs = xml.load("observation/y_obs_50GHz.xml")[:]

    # load true data
    atms = pa.xml.load("atmosphere/atmospheres_true.xml")
    atm = atms[226]

    y_true, jacobian_true = Forward_model(
        f_grid,
        atm,
        surface_reflectivity,
        surface_temperature,
        sensor_pos,
        sensor_los,
        retrieval_quantity="T",
    )

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
    ax[0].plot(f_grid / 1e9, y_true, "o")
    ax[0].set_ylabel("Brightness temperature [K]")
    ax[0].set_xlabel("Frequency [GHz]")

    ax[1].plot(jacobian.T, dropsonde.data[1, :, 0, 0] / 1e3)
    ax[1].set_prop_cycle(None)
    ax[1].plot(jacobian_true.T, atms[226][1, :, 0, 0] / 1e3, "--")
    ax[1].set_ylabel("Jacobian")
    ax[1].set_xlabel("Altitude [km]")
    ax[1].set_title("Temperature jacbian")
