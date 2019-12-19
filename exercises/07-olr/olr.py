#!/usr/bin/env python3
"""Simulate and plot the outgoing longwave radiation (OLR) and Planck curves
for different temperatures. """
import matplotlib.pyplot as plt
import numpy as np
import typhon as ty
import typhon.arts.workspace


ty.plots.styles.use()


def main():
    # Read input atmosphere
    atmfield = ty.arts.xml.load("input/midlatitude-summer.xml")

    # Scale the CO2 concentration
    atmfield.scale("abs_species-CO2", 1)
    atmfield.add("T", 0)

    # Calculate the outgoing-longwave radiation
    f, olr = calc_olr(atmfield)

    # Plotting.
    wn = ty.physics.frequency2wavenumber(f) / 100  # Hz -> cm^-1

    temps = [225, 250, 275, atmfield.get("T", keep_dims=False)[0]]
    temp_colors = typhon.plots.cmap2rgba("temperature", len(temps))

    fig, ax = plt.subplots()
    for t, color in sorted(zip(temps, temp_colors)):
        ax.plot(
            wn, np.pi * typhon.physics.planck(f, t), label=f"{t:3.1f} K", color=color
        )
    ax.plot(wn, olr, color="C0", label="Radiance")
    ax.legend()
    ax.set_title(rf"OLR={np.trapz(olr, f):3.2f} $\sf Wm^{{-2}}$")
    ax.set_xlim(wn.min(), wn.max())
    ax.set_xlabel(r"Wavenumber [$\sf cm^{-1}$]")
    ax.set_ylabel(r"Irradiance [$\sf Wm^{-2}Hz^{-1}$]")
    ax.set_ylim(bottom=0)
    fig.savefig("plots/olr.pdf")
    plt.show()


def calc_olr(atmfield, nstreams=2, fnum=300, fmin=1.0, fmax=75e12, verbosity=2):
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
    ws = ty.arts.workspace.Workspace(verbosity=0)
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

    # Read a line file and a matching small frequency grid
    ws.abs_linesReadFromSplitArtscat(
        ws.abs_species, "hitran/hitran_split_artscat5/", 0.9 * fmin, 1.1 * fmax
    )

    # Sort the line file according to species
    ws.abs_lines_per_speciesCreateFromLines()

    # Set the lineshape function for all calculated tags
    ws.abs_lineshapeDefine(shape="Voigt_Kuntz6", forefactor="VVH", cutoff=750e9)

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)
    ws.Copy(
        ws.surface_rtprop_agenda,
        ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface,
    )

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

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
    ws.doit_i_fieldClearskyPlaneParallel(trans_field=ws.trans_field, use_parallel_iy=1)
    ws.spectral_irradiance_fieldFromiyField()

    olr = ws.spectral_irradiance_field.value[:, -1, 0, 0, 1].copy()

    return ws.f_grid.value.copy(), olr


if __name__ == "__main__":
    main()
