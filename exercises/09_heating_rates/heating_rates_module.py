"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import numpy as np
import pyarts.workspace
from typhon import physics as phys


def calc_spectral_irradiance(atmfield,
                             nstreams=2,
                             fnum=300,
                             fmin=1.0,
                             fmax=97e12,
                             verbosity=0):
    """Calculate the spectral downward and upward irradiance for a given atmosphere.
    Irradiandce is defined as positive quantity independent of direction.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray, ndarray, ndarray, ndarray, ndarray :
        Frequency grid [Hz], altitude [m], pressure [Pa], temperature [K],
        spectral downward irradiance [Wm^-2 Hz^-1],
        spectral upward irradiance [Wm^-2 Hz^-1].
    """
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # (standard) emission calculation
    ws.iy_main_agendaSet(option="Emission")

    # cosmic background radiation
    ws.iy_space_agendaSet(option="CosmicBackground")

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    # sensor-only path
    ws.ppath_agendaSet(option="FollowSensorLosPath")

    # no refraction
    ws.ppath_step_agendaSet(option="GeometricPath")

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Definition of species
    ws.abs_speciesSet(species=[
        "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
        "CO2, CO2-CKDMT252"
    ])

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesNormalization(option=normalization)

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)
    ws.surface_rtprop_agendaSet(
        option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # No sensor properties
    ws.sensorOff()

    # Atmosphere and surface
    ws.atm_fields_compact = atmfield
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # Set surface height and temperature equal to the lowest atmosphere level
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    # Output radiance not converted
    ws.StringSet(ws.iy_unit, "1")

    # Definition of sensor position and LOS
    ws.MatrixSet(ws.sensor_pos, np.array([[100e3]]))  # sensor in z = 100 km
    ws.MatrixSet(ws.sensor_los,
                 np.array([[180]
                           ]))  # zenith angle: 0 looking up, 180 looking down

    # Perform RT calculations
    ws.propmat_clearsky_agendaAuto()
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.AngularGridsSetFluxCalc(N_za_grid=nstreams,
                               N_aa_grid=1,
                               za_grid_type="double_gauss")

    # calculate intensity field
    ws.Tensor3Create("trans_field")
    ws.spectral_radiance_fieldClearskyPlaneParallel(trans_field=ws.trans_field,
                                                    use_parallel_za=0)
    ws.spectral_irradiance_fieldFromSpectralRadianceField()

    spectral_flux_downward = -ws.spectral_irradiance_field.value[:, :, 0, 0,
                                                                 0].copy()
    spectral_flux_upward = ws.spectral_irradiance_field.value[:, :, 0, 0,
                                                              1].copy()

    spectral_flux_downward[np.isnan(spectral_flux_downward)] = 0.
    spectral_flux_upward[np.isnan(spectral_flux_upward)] = 0.

    # set outputs
    f = ws.f_grid.value[:].copy()
    z = ws.z_field.value[:].copy().squeeze()
    p = atmfield.grids[1][:].squeeze().copy()
    T = atmfield.get("T")[:].squeeze().copy()

    return f, z, p, T, spectral_flux_downward, spectral_flux_upward


def calc_irradiance(atmfield,
                    nstreams=2,
                    fnum=300,
                    fmin=1.0,
                    fmax=97e12,
                    verbosity=0):
    """Calculate the downward and upward irradiance for a given atmosphere.
    Irradiandce is defined as positive quantity independent of direction.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray, ndarray, ndarray, ndarray :
        Altitude [m], pressure [Pa], temperature [K],
        downward irradiance [Wm^-2], upward irradiance [Wm^-2].
    """

    f, z, p, T, spectral_flux_downward, spectral_flux_upward = calc_spectral_irradiance(
        atmfield,
        nstreams=nstreams,
        fnum=fnum,
        fmin=fmin,
        fmax=fmax,
        verbosity=verbosity)

    #calculate flux
    flux_downward = np.trapz(spectral_flux_downward, f, axis=0)
    flux_upward = np.trapz(spectral_flux_upward, f, axis=0)

    return z, p, T, flux_downward, flux_upward


def integrate_spectral_irradiance(f, spectral_flux, fmin=-np.inf, fmax=np.inf):
    """Calculate the integral of the spectral iradiance from fmin to fmax.

    Parameters:
        f (ndarray): Frequency grid [Hz].
        spectral_flux (ndarray): Spectral irradiance [Wm^-2 Hz^-1].
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].

    Returns:
        ndarray irradiance [Wm^-2].
    """

    logic = np.logical_and(fmin <= f, f < fmax)

    flux = np.trapz(spectral_flux[logic, :], f[logic], axis=0)

    return flux
