"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import numpy as np
import pyarts.workspace


def calc_olr(atmfield, nstreams=2, fnum=300, fmin=1.0, fmax=75e12, verbosity=0):
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

    # Definition of species
    ws.abs_speciesSet(
        species=[
            "H2O,H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
            "CO2, CO2-CKDMT252",
        ]
    )

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
       basename="spectroscopy/cat/"
    )

    # ws.abs_lines_per_speciesSetLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesSetCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesSetNormalization(option=normalization)

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)
    ws.Copy(
        ws.surface_rtprop_agenda,
        ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface,
    )

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
    ws.MatrixSet(
        ws.sensor_los, np.array([[180]])
    )  # zenith angle: 0 looking up, 180 looking down

    # Perform RT calculations
    ws.abs_xsec_agenda_checkedCalc()
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()
    ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.AngularGridsSetFluxCalc(
        N_za_grid=nstreams, N_aa_grid=1, za_grid_type="double_gauss"
    )

    # calculate intensity field
    ws.Tensor3Create("trans_field")
    ws.spectral_radiance_fieldClearskyPlaneParallel(
        trans_field=ws.trans_field, use_parallel_za=0
    )
    ws.spectral_irradiance_fieldFromSpectralRadianceField()

    olr = ws.spectral_irradiance_field.value[:, -1, 0, 0, 1].copy()

    return ws.f_grid.value.copy(), olr
