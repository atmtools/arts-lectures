"""Calculate and plot clear-sky Jacobians."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pyarts
import typhon as ty
from matplotlib.transforms import blended_transform_factory
from typhon.arts import xml


def argclosest(array, value):
    """Returns the index in ``array`` which is closest to ``value``."""
    return np.abs(array - value).argmin()


def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


def plot_brightness_temperature(frequency, y, where=None, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(frequency / 1e9, y)
    ax.set_xlim(frequency.min() / 1e9, frequency.max() / 1e9)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(r"$T\mathrm{_B}$ [K]")

    if where is not None:
        freq_ind = argclosest(frequency, where)
        l, = ax.plot(
            frequency[freq_ind] / 1e9, y[freq_ind], marker="o", color="tab:red"
        )
        ax.text(
            0.05,
            0.9,
            f"{frequency[freq_ind]/1e9:.2f} GHz",
            size="small",
            color=l.get_color(),
            transform=ax.transAxes,
        )


def plot_opacity(frequency, opacity, where=None, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.semilogy(frequency / 1e9, opacity[-1, :])
    ax.set_xlim(frequency.min() / 1e9, frequency.max() / 1e9)
    ax.axhline(1, color="ty:darkgrey", linewidth=0.8, zorder=-1)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Zenith Opacity")

    if where is not None:
        freq_ind = argclosest(frequency, where)
        ax.plot(
            frequency[freq_ind] / 1e9,
            opacity[-1, freq_ind],
            marker="o",
            color="tab:red",
        )


def plot_jacobian(height, jacobian, jacobian_quantity, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.semilogy(jacobian, height / 1000.0)
    ax.set_ylim(0.4, 70)
    unit = "K/K/km" if jacobian_quantity == "T" else "K/1/km"
    ax.set_xlabel(f"{tag2tex(jacobian_quantity)} Jacobian [{unit}]")
    ax.set_ylabel("$z$ [km]")
    jac_peak = height[np.abs(jacobian).argmax()] / 1000.0
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    lh = ax.axhline(jac_peak, color="ty:jetblack", zorder=3)
    ax.text(
        1,
        jac_peak,
        f"{jac_peak:.2f} km",
        size="small",
        ha="right",
        va="bottom",
        color=lh.get_color(),
        bbox={"color": "white", "alpha": 0.5},
        zorder=2,
        transform=trans,
    )


def plot_opacity_profile(height, opacity, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.loglog(opacity, height[::-1] / 1000.0)
    ax.set_xlim(1e-8, 1e2)
    ax.set_xticks(10.0 ** np.arange(-8, 4, 2))
    ax.set_xlabel(r"Opacity $\tau(z, z_\mathrm{TOA})$")
    ax.set_ylim(0.4, 70)
    ax.set_ylabel("$z$ [km]")

    try:
        tau1 = height[::-1][np.where(opacity >= 1)[0][0]]
    except IndexError:
        pass
    else:
        tau1 /= 1000
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        lh = ax.axhline(tau1, color="ty:jetblack", zorder=3)
        ax.text(
            0.05,
            tau1,
            f"{tau1:.2f} km",
            va="bottom",
            size="small",
            color=lh.get_color(),
            bbox={"color": "white", "alpha": 0.5},
            zorder=2,
            transform=trans,
        )
        ax.axvline(1, color="ty:darkgrey", linewidth=0.8, zorder=-1)


def calc_jacobians(
    jacobian_quantity="H2O", fmin=150e9, fmax=200e9, fnum=200, verbosity=2
):
    """Calculate jacobians for a given species and frequency range."""
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Agenda for scalar gas absorption calculation
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    # Modified emission agenda to store internally calculated optical thickness.
    @pyarts.workspace.arts_agenda
    def iy_main_agenda__EmissionOpacity(ws):
        ws.ppathCalc()
        ws.iyEmissionStandard()
        ws.ppvar_optical_depthFromPpvar_trans_cumulat()
        ws.WriteXML("ascii", ws.ppvar_optical_depth, "results/optical_thickness.xml")
        ws.WriteXML("ascii", ws.ppvar_p, "results/ppvar_p.xml")

    ws.Copy(ws.iy_main_agenda, iy_main_agenda__EmissionOpacity)

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

    #########################################################################

    # A pressure grid rougly matching 0 to 80 km, in steps of 2 km.
    ws.VectorNLogSpace(ws.p_grid, 200, 1013e2, 10.0)

    # Definition of species:
    # you can take out and add again one of the species to see what effect it has
    # on radiative transfer in the ws.atmosphere.
    abs_species = {"N2", "O2", "H2O"}
    if jacobian_quantity != "T":
        abs_species.add(jacobian_quantity)

    ws.abs_speciesSet(species=list(abs_species))

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
       basename="spectroscopy/Hitran/"
    )
    
    # ws.abs_lines_per_speciesSetLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesSetCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesSetNormalization(option=normalization)
    
    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric scenario
    ws.AtmRawRead(basename="planets/Earth/Fascod/midlatitude-summer/midlatitude-summer")

    # Non reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.4)
    ws.Copy(
        ws.surface_rtprop_agenda,
        ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface,
    )

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    #########################################################################

    # Atmosphere and surface
    ws.AtmosphereSet1D()
    ws.AtmFieldsCalc(interp_order=3)
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    # Definition of sensor position and line of sight (LOS)
    ws.VectorSet(ws.rte_pos, np.array([800e3]))
    ws.MatrixSet(ws.sensor_pos, np.array([[800e3]]))
    ws.MatrixSet(ws.sensor_los, np.array([[180]]))
    ws.VectorSet(ws.rte_los, np.array([180]))
    ws.sensorOff()

    # Jacobian calculation
    ws.jacobianInit()
    if jacobian_quantity == "T":
        ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
    else:
        ws.jacobianAddAbsSpecies(
            g1=ws.p_grid,
            g2=ws.lat_grid,
            g3=ws.lon_grid,
            species=jacobian_quantity,
            unit="rel",
        )
    ws.jacobianClose()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Perform RT calculations
    ws.abs_xsec_agenda_checkedCalc()
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.yCalc()

    # Write output
    ws.WriteXML("ascii", ws.f_grid, "results/f_grid.xml")
    ws.WriteXML("ascii", ws.jacobian, "results/jacobian.xml")
    ws.WriteXML("ascii", ws.z_field, "results/z_field.xml")
    ws.WriteXML("ascii", ws.y, "results/y.xml")
