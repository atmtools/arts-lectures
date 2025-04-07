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


import numpy as np
import pyarts.workspace
import pyarts as pa
from pyarts import xml

# %% ARTS functions

def basic_setup(f_grid, sensor_description=[], verbosity=0):
    """
    Sets up a basic ARTS workspace configuration for radiative transfer calculations.
    This function initializes an ARTS workspace with standard settings for atmospheric
    radiative transfer calculations, particularly focused on microwave sensors and
    tropospheric applications.
    Parameters
    ----------
    f_grid : numpy.ndarray
        Frequency grid for calculations. Must be provided if sensor_description is not used.
    sensor_description : list, optional
        AMSU sensor description parameters. Cannot be used simultaneously with f_grid.
        Default is empty list.
    version : str, optional
        Version of ARTS catalogue to be downloaded. Default is "2.6.8".
    verbosity : int, optional
        Level of output verbosity. Default is 0 (minimal output).
    Returns
    -------
    ws : pyarts.workspace.Workspace
        Configured ARTS workspace instance.
    Raises
    ------
    ValueError
        If both f_grid and sensor_description are provided simultaneously,
        or if neither f_grid nor sensor_description is provided.
    Notes
    -----
    The function sets up:
    - Standard emission calculation
    - Cosmic background radiation
    - Surface properties (non-reflecting surface)
    - Absorption species (H2O, O2, N2 using Rosenkranz models)
    - Planck brightness temperature as output unit
    - 1D atmosphere
    - Path calculation without refraction
    For sensor description, the function includes an iterative adjustment mechanism
    that modifies the frequency spacing if initial sensor response generation fails.
    """


    pa.cat.download.retrieve(verbose=bool(verbosity))

    ws = pa.workspace.Workspace(verbosity=verbosity)
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

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    ws.AtmosphereSet1D()

    it_max=5

    if np.size(f_grid)==0 and np.size(sensor_description)>0:
        iterate=True
        N_it=0
        while iterate and N_it<it_max:
            N_it+=1
            try:
                ws.sensor_description_amsu=sensor_description
                ws.sensor_responseGenericAMSU(spacing=1e12)
                iterate=False
            except RuntimeError:
                rel_change=0.9

                print(f'adjusting relative mandatory minimum frequency spacing by factor {rel_change}')

                #adjust relative mandatory minimum frequency spacing
                sensor_description[:,-1]*=rel_change


    elif np.size(f_grid)>0 and np.size(sensor_description)>0:

        raise ValueError("f_grid and sensor_description cannot be provided simultaneously")

    elif np.size(f_grid)>0 and np.size(sensor_description)==0:

        # Set the frequency grid
        ws.f_grid = f_grid
        ws.sensorOff()
    else:
        raise ValueError("f_grid or sensor_description must be provided")

    ws.jacobianOff()

    # on-the-fly absorption
    ws.propmat_clearsky_agendaAuto()
    ws.abs_lines_per_speciesSetEmpty()

    #switch off jacobian calculation by default
    ws.jacobianOff()

    return ws


def forward_model(
    ws,
    atm_fields_compact,
    surface_reflectivity,
    surface_temperature,
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
    else:
        ws.jacobianOff()

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
    sensor_altitude,
    sensor_los,
    retrieval_quantity="",
    sensor_description=[],
    verbose=False,
):
    """
    Forward radiative transfer model for atmospheric observations.
    This function computes the radiative transfer and optionally its Jacobian for given
    atmospheric conditions and sensor specifications.
    Parameters
    ----------
    f_grid : array-like
        Frequency grid for calculations [Hz]. Ignored if sensor_description is provided.
    atm_fields_compact : dict
        Dictionary containing atmospheric fields (temperature, humidity etc.)
    surface_reflectivity : float or array-like
        Surface reflectivity value(s)
    surface_temperature : float
        Surface temperature [K]
    sensor_altitude : float
        Altitude of the sensor [m]
    sensor_los : float
        Line of sight angle of the sensor [degrees]
    retrieval_quantity : str, optional
        Quantity for which to calculate Jacobian. Default is empty string.
    sensor_description : list, optional
        Sensor channel specifications. If provided, f_grid is ignored.
    Returns
    -------
    tuple
        - y : array-like
            Simulated radiances/brightness temperatures
        - jacobian : array-like
            Jacobian matrix if retrieval_quantity is specified, otherwise None
    Notes
    -----
    The function uses a workspace (ws) with either frequency grid or sensor description
    for radiative transfer calculations.
    """

    if np.size(sensor_description)>0:
        if verbose:
            print('Ignoring f_grid\n output will be sensor channels')
        ws = basic_setup([],sensor_description=sensor_description, verbosity=verbose)
    else:
        ws = basic_setup(f_grid, verbosity=verbose)

    set_sensor_position_and_view(ws, sensor_altitude, sensor_los)

    y, jacobian = forward_model(
        ws,
        atm_fields_compact,
        surface_reflectivity,
        surface_temperature,
        retrieval_quantity=retrieval_quantity,
    )


    return y, jacobian


def set_sensor_position_and_view(ws, sensor_pos, sensor_los):
    """Set sensor position and line-of-sight direction in workspace.
    This function sets the sensor position and line-of-sight (viewing direction)
    in the workspace for radiative transfer calculations.
    Parameters
    ----------
    ws : Workspace
        ARTS workspace object where sensor parameters will be set
    sensor_pos : array-like
        Sensor position coordinates (e.g., [x,y,z])
    sensor_los : array-like
        Line-of-sight direction vector (e.g., [dx,dy,dz])
    Returns
    -------
    None
        Modifies workspace in-place by setting sensor_pos and sensor_los variables
    """

    ws.sensor_pos=np.array([[sensor_pos]])
    ws.sensor_los=np.array([[sensor_los]])


def prepare_initial_conditions(
    ws,
    atmosphere_apr,
    surface_temperature,
    surface_reflectivity,
    O2vmr=0.2095,
    N2vmr=0.7808,
):
    """Prepare initial atmospheric conditions for radiative transfer calculations.
    This function sets up the atmospheric state, including the addition of N2 and O2
    if not present in the atmospheric data, and configures surface properties.
    Parameters
    ----------
    ws : Workspace
        ARTS workspace object.
    atmosphere_apr : GriddedField4
        A priori atmospheric state containing species and their concentrations.
    surface_temperature : float
        Surface temperature in Kelvin.
    surface_reflectivity : float or list
        Surface reflectivity value(s).
    O2vmr : float, optional
        Volume mixing ratio of O2. Defaults to 0.2095.
    N2vmr : float, optional
        Volume mixing ratio of N2. Defaults to 0.7808.
    Notes
    -----
    The function performs the following operations:
    1. Sets the atmospheric fields from the input atmosphere
    2. Adds N2 and O2 if not present in the atmosphere
    3. Converts compact fields to full atmospheric fields
    4. Sets surface temperature and reflectivity
    """

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
    sensor_description=[],
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

    if np.size(sensor_description)>0:
        print('Ignoring f_grid\n output will be sensor channels')
        ws = basic_setup([],sensor_description=sensor_description)
    else:
        ws = basic_setup(f_grid)

    prepare_initial_conditions(
        ws, background_atmosphere, surface_temperature, surface_reflectivity
    )

    set_sensor_position_and_view(ws, sensor_altitude, sensor_los)

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


def Hamp_channels(band_selection, rel_mandatory_grid_spacing=1./4.):
    """
    Returns sensor description and characteristics for HAMP (Humidity And Temperature Profiler) channels.

    This function provides frequency specifications and sensor characteristics for different
    frequency bands (K, V, W, F, G) of the HAMP instrument. Each band contains multiple channels
    with specific center frequencies, offsets, and other parameters.

    Parameters
    ----------
    band_selection : list
        List of strings indicating which frequency bands to include.
        Valid options are 'K', 'V', 'W', 'F', and 'G'.
        If empty list is provided, prints available bands and their specifications.
    rel_mandatory_grid_spacing : float, optional
        Relative mandatory frequency grid spacing for the passbands.
        Default is 0.25 (1/4). This means that the mandatory grid spacing is 1/4 of the
        passbands bandwidth.

    Returns
    -------
    tuple or None
        If band_selection is empty, returns None and prints available bands.
        Otherwise returns tuple of (sensor_description, NeDT, Accuracy, FWHM_Antenna):
            - sensor_description : ndarray
                Array of [frequency, offset1, offset2, bandwidth, df] for each channel
            - NeDT : ndarray
                Noise equivalent differential temperature for each channel
            - Accuracy : ndarray
                Accuracy in Kelvin for each channel
            - FWHM_Antenna : ndarray
                Full Width at Half Maximum of antenna beam pattern in degrees

    Raises
    ------
    ValueError
        If an invalid band is specified in band_selection.

    Notes
    -----
    Frequency bands:
    - K band: 7 channels around 22-31 GHz
    - V band: 7 channels around 50-58 GHz
    - W band: 1 channel at 90 GHz
    - F band: 4 channels around 118.75 GHz
    - G band: 6 channels around 183.31 GHz
    """



    channels = {}
    channels["K"] = {
        "f_center": np.array([22.24, 23.04, 23.84, 25.44, 26.24, 27.84, 31.40]) * 1e9,
        "Offset1": np.zeros(7),
        "Offset2": np.zeros(7),
        "NeDT": 0.1,
        "Accuracy": 0.5,
        "Bandwidth": 230e6,
        "FWHM_Antenna": 5.0,
        "df": rel_mandatory_grid_spacing,
    }
    channels["V"] = {
        "f_center": np.array([50.3, 51.76, 52.8, 53.75, 54.94, 56.66, 58.00]) * 1e9,
        "Offset1": np.zeros(7),
        "Offset2": np.zeros(7),
        "NeDT": 0.2,
        "Accuracy": 0.5,
        "Bandwidth": 230e6,
        "FWHM_Antenna": 3.5,
        "df": rel_mandatory_grid_spacing,
    }
    channels["W"] = {
        "f_center": np.array([90])* 1e9,
        "Offset1": np.zeros(1),
        "Offset2": np.zeros(1),
        "NeDT": 0.25,
        "Accuracy": 1.5,
        "Bandwidth": 2e9,
        "FWHM_Antenna": 3.3,
        "df": rel_mandatory_grid_spacing,
    }

    channels["F"] = {
        "f_center": np.ones(4) * 118.75e9,
        "Offset1": np.array([1.3, 2.3, 4.2, 8.5]) * 1e9,
        "Offset2": np.zeros(4),
        "NeDT": 0.6,
        "Accuracy": 1.5,
        "Bandwidth": 400e6,
        "FWHM_Antenna": 3.3,
        "df": rel_mandatory_grid_spacing,

    }

    channels["G"] = {
        "f_center": np.ones(6) * 183.31e9,
        "Offset1": np.array([0.6, 1.5, 2.5, 3.5, 5.0, 7.5]) * 1e9,
        "Offset2": np.zeros(6),
        "NeDT": 0.6,
        "Accuracy": 1.5,
        "Bandwidth": np.array([200e6, 200e6, 200e6, 200e6, 200e6, 1000e6]),
        "FWHM_Antenna": 2.7,
        "df": rel_mandatory_grid_spacing,
    }


    if len(band_selection) == 0:
        print("No band selected")
        print("Following bands are available:\n")
        for key in channels.keys():
            print(f'{key} =====================================================')
            print(f'f_grid: {channels[key]["f_center"]} Hz')
            print(f'Offset1: {channels[key]["Offset1"]} Hz')
            print(f'Offset2: {channels[key]["Offset2"]} Hz')
            print(f'NeDT: {channels[key]["NeDT"]} K')
            print(f'Accuracy: {channels[key]["Accuracy"]} K')
            print(f'Bandwidth: {channels[key]["Bandwidth"]} Hz')
            print(f'FWHM_Antenna: {channels[key]["FWHM_Antenna"]} deg')
            print(f'df: {channels[key]["df"]}')
            print('=====================================================\n')
        return



    else:

        sensor_description = []
        NeDT = []
        Accuracy = []
        FWHM_Antenna = []

        for band in band_selection:

            if band in channels.keys():

                for i in range(len(channels[band]["f_center"])):
                    freq = channels[band]["f_center"][i]
                    offset1 = channels[band]["Offset1"][i]
                    offset2 = channels[band]["Offset2"][i]

                    if isinstance(channels[band]["Bandwidth"], float):
                        bandwidth = channels[band]["Bandwidth"]
                    else:
                        bandwidth = channels[band]["Bandwidth"][i]

                    desc_i = [freq, offset1, offset2, bandwidth, bandwidth*channels[band]["df"]]

                    sensor_description.append(desc_i)
                    NeDT.append(channels[band]["NeDT"])
                    Accuracy.append(channels[band]["Accuracy"])
                    FWHM_Antenna.append(channels[band]["FWHM_Antenna"])

            else:
                raise   ValueError(f"Band {band} not available")

        return np.array(sensor_description), np.array(NeDT), np.array(Accuracy), np.array(FWHM_Antenna)


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

    fig.tight_layout()
