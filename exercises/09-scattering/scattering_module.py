"""Perform a DOIT scattering radiation with ARTS.

Based on a script by Jakob Doerr.
"""
import numpy as np
import pyarts


"function taken from typhon.physics"


def radiance2planckTb(f, r):
    """Convert spectral radiance to Planck brightness temperture.

    Parameters:
        f (float or ndarray): Frequency [Hz].
        r (float or ndarray): Spectral radiance [W/(m2*Hz*sr)].

    Returns:
        float or ndarray: Planck brightness temperature [K].
    """
    c = pyarts.arts.constants.c
    k = pyarts.arts.constants.k
    h = pyarts.arts.constants.h

    return h / k * f / np.log(np.divide((2 * h / c**2) * f**3, r) + 1)


def argclosest(array, value, retvalue=False):
    """Returns the index of the closest value in array."""
    idx = np.abs(array - value).argmin()

    return (idx, array[idx]) if retvalue else idx


def setup_doit_agendas(ws):
    # Calculation of the phase matrix
    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def pha_mat_spt_agenda(ws):
        # Optimized option:
        ws.pha_mat_sptFromDataDOITOpt()
        # Alternative option:
        # ws.pha_mat_sptFromMonoData

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def doit_scat_field_agenda(ws):
        ws.doit_scat_fieldCalcLimb()
        # Alternative: use the same za grids in RT part and scattering integral part
        # ws.doit_scat_fieldCalc()

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def doit_rte_agenda(ws):
        # Sequential update for 1D
        ws.cloudbox_fieldUpdateSeq1D()

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def spt_calc_agenda(ws):
        ws.opt_prop_sptFromMonoData()

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def doit_conv_test_agenda(ws):
        ws.doit_conv_flagAbsBT(
            epsilon=np.array([0.01]), max_iterations=100, nonconv_return_nan=1
        )
        ws.Print(ws.doit_iteration_counter, 0)

    ws.doit_conv_test_agenda = doit_conv_test_agenda

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def doit_mono_agenda(ws):
        # Prepare scattering data for DOIT calculation (Optimized method):
        ws.Ignore(ws.f_grid)
        ws.DoitScatteringDataPrepare()
        # Perform iterations: 1. scattering integral. 2. RT calculations with
        # fixed scattering integral field, 3. convergence test
        ws.cloudbox_field_monoIterate(accelerated=1)

    ws.iy_cloudbox_agendaSet(option="LinInterpField")


def scattering(
    ice_water_path=2.0, num_viewing_angles=19, phase="ice", radius=1.5e2, verbosity=0
):
    """Perform a radiative transfer simulation with a simple cloud.

    Parameters:
        ice_water_path (float): Integrated ice water in cloud box [kg/m^2].
        num_viewing_angles (int): Number of zenith viewing angles.
        phase (str): Hydrometeor phase "ice" or "liquid".
        radius (float): Particle radius.
        verbosity (int): Reporting levels between 0 (only error messages)
                    and 3 (everything).

    Returns:
        ndarray, ndarray, ndarray, ndarray:
            Pressure [hPa],
            Viewing angles [degree],
            Radiation field [K],
            Radiation field clear-sky [K].

    """

    pyarts.cat.download.retrieve(verbose=bool(verbosity))

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Agenda settings

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

    ws.VectorSet(ws.f_grid, np.array([229.0e9]))  # Define f_grid

    ws.IndexSet(ws.stokes_dim, 1)  # Set stokes dim

    ws.AtmosphereSet1D()

    ws.jacobianOff()  # No jacobian calculations

    ws.sensorOff()  # No sensor

    # Set the maximum propagation step to 250m.
    ws.NumericSet(ws.ppath_lmax, 250.0)

    # Set absorption species
    ws.abs_speciesSet(
        species=["H2O-PWR98", "O3", "O2-PWR98", "N2-SelfContStandardType"]
    )

    # Read atmospheric data
    ws.ReadXML(ws.batch_atm_fields_compact, "input/chevallierl91_all_extract.xml")
    ws.Extract(ws.atm_fields_compact, ws.batch_atm_fields_compact, 1)

    # Add constant profiles for O2 and N2
    ws.atm_fields_compactAddConstant(
        name="abs_species-O2", value=0.2095, condensibles=["abs_species-H2O"]
    )
    ws.atm_fields_compactAddConstant(
        name="abs_species-N2", value=0.7808, condensibles=["abs_species-H2O"]
    )

    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
    ws.atmfields_checkedCalc()

    # Read Catalog (needed for O3):
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesNormalization(option=normalization)

    # absorption from LUT
    ws.propmat_clearsky_agendaAuto()

    ws.abs_lookupSetup()
    ws.lbl_checkedCalc()

    ws.abs_lookupCalc()

    ws.propmat_clearsky_agenda_checkedCalc()

    # Set surface reflectivity (= 1 - emissivity)
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    # Extract particle mass from scattering meta data.
    scat_xml = f"scattering/H2O_{phase}/MieSphere_R{radius:.5e}um"
    scat_meta = pyarts.arts.ScatteringMetaData()
    scat_meta.readxml(scat_xml + ".meta")
    particle_mass = float(scat_meta.mass)

    # Load scattering data and PND field.
    ws.ScatSpeciesInit()
    ws.ScatElementsPndAndScatAdd(
        scat_data_files=[scat_xml], pnd_field_files=["./input/pndfield_input.xml"]
    )
    ws.scat_dataCalc()
    ws.scat_data_checkedCalc()

    # Set the extent of the cloud box.
    ws.cloudboxSetManually(
        p1=101300.0, p2=1000.0, lat1=0.0, lat2=0.0, lon1=0.0, lon2=0.0
    )

    # Trim pressure grid to match cloudbox.
    bottom, top = ws.cloudbox_limits.value
    p = ws.p_grid.value[bottom : top + 1].copy()
    z = ws.z_field.value[bottom : top + 1, 0, 0].copy()

    ws.pnd_fieldCalcFrompnd_field_raw()

    # Calculate the initial ice water path (IWP).
    iwp0 = np.trapz(particle_mass * ws.pnd_field.value[0, :, 0, 0], z)

    # Scale the PND field to get the desired IWP.
    ws.Tensor4Multiply(ws.pnd_field, ws.pnd_field, ice_water_path / iwp0)

    # Get case-specific surface properties from corresponding atmospheric fields
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    # Consistency checks
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()

    setup_doit_agendas(ws)

    ws.doit_za_interpSet(interp_method="linear")

    ws.DOAngularGridsSet(
        N_za_grid=num_viewing_angles, N_aa_grid=37, za_grid_opt_file=""
    )

    # Use lookup table for absorption calculation
    ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

    # Scattering calculation
    ws.DoitInit()
    ws.DoitGetIncoming(rigorous=0)
    ws.cloudbox_fieldSetClearsky()
    ws.DoitCalc()

    ifield = np.squeeze(ws.cloudbox_field.value[:].squeeze())
    ifield = radiance2planckTb(ws.f_grid.value, ifield)

    # Clear-sky
    ws.Tensor4Multiply(ws.pnd_field, ws.pnd_field, 0.0)
    ws.DoitCalc()

    ifield_clear = np.squeeze(ws.cloudbox_field.value[:].squeeze())
    ifield_clear = radiance2planckTb(ws.f_grid.value, ifield_clear)

    return p, ws.za_grid.value[:].copy(), ifield, ifield_clear
