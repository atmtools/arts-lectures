"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import numpy as np
import pyarts.workspace


def calc_spectral_irradiance(
    atmfield, nstreams=4, fnum=300, fmin=1.0, fmax=97e12, verbosity=0
):
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

    pyarts.cat.download.retrieve(verbose=bool(verbosity))
    
    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculations
    ws.jacobianOff()

    # Definition of species
    ws.abs_speciesSet(
        species=[
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "CO2, CO2-CKDMT252",
        ]
    )

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read cross section data
    ws.ReadXsecData(basename="lines/")

    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Calculate absorption
    ws.propmat_clearsky_agendaAuto()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

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

    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    # Perform RT calculations
    ws.spectral_irradiance_fieldDisort(nstreams=nstreams, emission=1)

    spectral_flux_downward = -ws.spectral_irradiance_field.value[:, :, 0, 0, 0].copy()
    spectral_flux_upward = ws.spectral_irradiance_field.value[:, :, 0, 0, 1].copy()

    # spectral_flux_downward[np.isnan(spectral_flux_downward)] = 0.
    # spectral_flux_upward[np.isnan(spectral_flux_upward)] = 0.

    # set outputs
    f = ws.f_grid.value[:].copy()
    z = ws.z_field.value[:].copy().squeeze()
    p = atmfield.grids[1][:].squeeze().copy()
    T = atmfield.get("T")[:].squeeze().copy()

    return f, z, p, T, spectral_flux_downward, spectral_flux_upward


def calc_irradiance(atmfield, nstreams=2, fnum=300, fmin=1.0, fmax=97e12, verbosity=0):
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
        verbosity=verbosity,
    )

    # calculate flux
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

    flux = np.trapezoid(spectral_flux[logic, :], f[logic], axis=0)

    return flux
