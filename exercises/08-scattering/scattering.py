"""Perform a DOIT scattering radiation with ARTS.

Based on a script by Jakob Doerr.
"""
import numpy as np
import matplotlib.pyplot as plt
import typhon as ty
import typhon.arts.workspace
from matplotlib.ticker import StrMethodFormatter


def main():
    # Control parameters
    zenith_angle = 180.0  # viewing angle [degree, 180° = upward radiation]
    pressure_level = None  # pressure level [Pa]

    # Run ARTS simulation
    p, zenith_angles, ifield, ifield_clearsky = scattering()

    # Plot Tb vs height for a specific viewing angle
    ty.plots.styles.use("typhon")

    ia, zenith_angle = argclosest(zenith_angles, zenith_angle, retvalue=True)

    f0, a0 = plt.subplots()
    a0.plot(ifield_clearsky[:, ia], p / 100, label="Clear-sky")
    a0.plot(ifield[:, ia], p / 100, label="Scattering")
    a0.grid()
    a0.set_ylim(p.max() / 100, p.min() / 100)
    a0.set_ylabel("Pressure [hPa]")
    a0.set_xlabel(r"$T_\mathrm{B}$ [K]")
    a0.legend()
    a0.set_title(rf"$T_\mathrm{{B}}$ at $\Theta$ = {zenith_angle:.0f}°")

    # Plot Tb vs Viewing angle for a specific pressure level:
    if pressure_level is not None:
        ip, pressure_level = argclosest(p, pressure_level, retvalue=True)

        f1, a1 = plt.subplots(subplot_kw=dict(projection="polar"))
        a1.plot(np.deg2rad(zenith_angles), ifield_clearsky[ip, :], label="Clear-sky")
        a1.plot(np.deg2rad(zenith_angles), ifield[ip, :], label="Scattering")
        a1.legend(loc="upper right")
        a1.set_theta_offset(np.deg2rad(+90))
        a1.set_theta_direction(-1)
        a1.set_thetagrids(np.arange(0, 181, 45), ha="left")
        a1.text(0.01, 0.75, r"$T_\mathrm{B}$", transform=a1.transAxes)
        a1.yaxis.set_major_formatter(StrMethodFormatter("{x:g} K"))
        a1.set_thetamin(0)
        a1.set_thetamax(180)
        a1.set_xlabel(r"Viewing angle $\Theta$")
        a1.set_title(rf"$T_\mathrm{{B}}$ at p = {pressure_level/100:.0f} hPa")
    plt.show()


def argclosest(array, value, retvalue=False):
    """Returns the index of the closest value in array.  """
    idx = np.abs(array - value).argmin()

    return (idx, array[idx]) if retvalue else idx


def scattering(
    ice_water_path=2.0, num_viewing_angles=37, phase="ice", radius=1.5e2, verbosity=2
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
    ws = ty.arts.workspace.Workspace(verbosity=0)
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/agendasDOIT.arts")
    ws.execute_controlfile("general/planet_earth.arts")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Agenda settings

    # Agenda for scalar gas absorption calculation
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    # (standard) emission calculation
    ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)

    # cosmic background radiation
    ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)

    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)

    # no refraction
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)

    # absorption from LUT
    ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__LookUpTable)

    ws.VectorSet(ws.f_grid, np.array([229.0e9]))  # Define f_grid

    ws.IndexSet(ws.stokes_dim, 1)  # Set stokes dim

    ws.AtmosphereSet1D()

    ws.jacobianOff()  # No jacobian calculations

    ws.sensorOff()  # No sensor

    # Set the maximum propagation step to 250m.
    ws.NumericSet(ws.ppath_lmax, 250.0)

    # Set absorption species
    ws.abs_speciesSet(
        species=["H2O-PWR98", "O3", "O2-PWR93", "N2-SelfContStandardType"]
    )

    # Read atmospheric data
    ws.ReadXML(ws.batch_atm_fields_compact, "testdata/chevallierl91_all_extract.xml")
    ws.Extract(ws.atm_fields_compact, ws.batch_atm_fields_compact, 1)

    # Add constant profiles for O2 and N2
    ws.atm_fields_compactAddConstant(
        name="abs_species-O2", value=0.2095, condensibles=["abs_species-H2O"]
    )
    ws.atm_fields_compactAddConstant(
        name="abs_species-N2", value=0.7808, condensibles=["abs_species-H2O"]
    )

    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
    ws.atmfields_checkedCalc(bad_partition_functions_ok=1)

    # Read Catalog (needed for O3):
    ws.ReadSplitARTSCAT(
        abs_species=ws.abs_species,
        basename="hitran/hitran_split_artscat5/",
        fmin=0.9 * ws.f_grid.value.min(),
        fmax=1.1 * ws.f_grid.value.max(),
        globalquantumnumbers="",
        localquantumnumbers="",
        ignore_missing=0,
    )
    ws.abs_lines_per_speciesCreateFromLines()

    ws.abs_lookupSetup()
    ws.abs_xsec_agenda_checkedCalc()
    ws.lbl_checkedCalc()

    ws.abs_lookupCalc()

    ws.propmat_clearsky_agenda_checkedCalc()

    # Set surface reflectivity (= 1 - emissivity)
    ws.Copy(
        ws.surface_rtprop_agenda,
        ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface,
    )
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    # Extract particle mass from scattering meta data.
    scat_data_xml = f"scattering/H2O_{phase}/MieSphere_R{radius:.5e}um.xml"
    ws.ScatSpeciesScatAndMetaRead(scat_data_files=[scat_data_xml])
    particle_mass = ws.scat_meta.value[0][0].mass

    # Load scattering data and PND field.
    ws.ScatSpeciesInit()
    ws.ScatElementsPndAndScatAdd(
        scat_data_files=[scat_data_xml], pnd_field_files=["./input/pndfield_input.xml"]
    )
    ws.scat_dataCalc()
    ws.scat_data_checkedCalc()

    # Set the extent of the cloud box.
    ws.cloudboxSetManually(
        p1=101300.0, p2=1000.0, lat1=0.0, lat2=0.0, lon1=0.0, lon2=0.0
    )

    # Trim pressure grid to match cloudbox.
    bottom, top = ws.cloudbox_limits.value
    p = ws.p_grid.value[bottom : top + 1]
    z = ws.z_field.value[bottom : top + 1, 0, 0]

    ws.pnd_fieldCalcFrompnd_field_raw()

    # Calculate the initial ice water path (IWP).
    iwp0 = np.trapz(particle_mass * ws.pnd_field.value[0, :, 0, 0], z)

    # Scale the PND field to get the desired IWP.
    ws.Tensor4Scale(ws.pnd_field, ws.pnd_field, ice_water_path / iwp0)

    # Get case-specific surface properties from corresponding atmospheric fields
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    # Consistency checks
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()

    @ty.arts.workspace.arts_agenda
    def doit_conv_test_agenda(ws):
        ws.doit_conv_flagAbsBT(
            epsilon=np.array([0.01]), max_iterations=100, nonconv_return_nan=1
        )
        ws.Print(ws.doit_iteration_counter, level=0)

    ws.Copy(ws.doit_conv_test_agenda, doit_conv_test_agenda)

    ws.doit_za_interpSet(interp_method="linear")

    ws.DOAngularGridsSet(num_viewing_angles, 19, "")

    @ty.arts.workspace.arts_agenda
    def doit_mono_agenda(ws):
        # Prepare scattering data for DOIT calculation (Optimized method):
        ws.Ignore(ws.f_grid)
        ws.DoitScatteringDataPrepare()
        # Perform iterations: 1. scattering integral. 2. RT calculations with
        # fixed scattering integral field, 3. convergence test
        ws.cloudbox_field_monoIterate(accelerated=1)

    ws.Copy(ws.doit_mono_agenda, doit_mono_agenda)

    # Scattering calculation
    ws.DoitInit()
    ws.DoitGetIncoming(rigorous=0)
    ws.cloudbox_fieldSetClearsky()
    ws.DoitCalc()

    ifield = np.squeeze(ws.cloudbox_field.value.squeeze())
    ifield = ty.physics.radiance2planckTb(ws.f_grid.value, ifield)

    # Clear-sky
    ws.Tensor4Scale(ws.pnd_field, ws.pnd_field, 0.0)
    ws.DoitCalc()

    ifield_clear = np.squeeze(ws.cloudbox_field.value.squeeze())
    ifield_clear = ty.physics.radiance2planckTb(ws.f_grid.value, ifield_clear)

    return p, ws.za_grid.value.copy(), ifield, ifield_clear


if __name__ == "__main__":
    main()
