# %% Import modules and define functions
"""Calculate and plot clear-sky Jacobians."""
import re
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import pyarts
from matplotlib.transforms import blended_transform_factory


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
        (l,) = ax.plot(
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
    ax.axhline(1, color="darkgrey", linewidth=0.8, zorder=-1)
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

    ax.plot(jacobian, height / 1000.0)
    ax.set_ylim(0.4, 20)
    unit = "K/K/km" if jacobian_quantity == "T" else "K/1/km"
    ax.set_xlabel(f"{tag2tex(jacobian_quantity)} Jacobian [{unit}]")
    ax.set_ylabel("$z$ [km]")
    jac_peak = height[np.abs(jacobian).argmax()] / 1000.0
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    lh = ax.axhline(jac_peak, color="black", zorder=3)
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

    ax.semilogx(opacity, height[::-1] / 1000.0)
    ax.set_xlim(1e-8, 1e2)
    ax.set_xticks(10.0 ** np.arange(-8, 4, 2))
    ax.set_xlabel(r"Opacity $\tau(z, z_\mathrm{TOA})$")
    ax.set_ylim(0.4, 20)
    ax.set_ylabel("$z$ [km]")

    try:
        tau1 = height[::-1][np.where(opacity >= 1)[0][0]]
    except IndexError:
        pass
    else:
        tau1 /= 1000
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        lh = ax.axhline(tau1, color="black", zorder=3)
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
        ax.axvline(1, color="darkgrey", linewidth=0.8, zorder=-1)


def calc_jacobians(
    jacobian_quantity="H2O", fmin=150e9, fmax=200e9, fnum=200, verbosity=0,
):
    """Calculate jacobians for a given species and frequency range and
       save the result as arts-xml-files.

    Parameters:
        jacobian_quantity (str): Species tag for which to calculate the
            jacobian.
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.
        verbosity (int): ARTS verbosity level.

    """

    pyarts.cat.download.retrieve(verbose=bool(verbosity))
    makedirs("results", exist_ok=True)

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Modified emission agenda to store internally calculated optical thickness.
    @pyarts.workspace.arts_agenda
    def iy_main_agenda__EmissionOpacity(ws):
        ws.ppathCalc()
        ws.iyEmissionStandard()
        ws.ppvar_optical_depthFromPpvar_trans_cumulat()
        ws.Touch(ws.geo_pos)
        ws.WriteXML("ascii", ws.ppvar_optical_depth, "results/optical_thickness.xml")
        ws.WriteXML("ascii", ws.ppvar_p, "results/ppvar_p.xml")

    ws.iy_main_agenda = iy_main_agenda__EmissionOpacity

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
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric scenario
    ws.AtmRawRead(basename="planets/Earth/Fascod/midlatitude-summer/midlatitude-summer")

    # Non reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.4)
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

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
    ws.propmat_clearsky_agendaAuto()
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


# %% Calculate and plot Jacobians
if __name__ == "__main__":
    # Calculate Jacobians (ARTS)
    jacobian_quantity = "H2O"
    calc_jacobians(jacobian_quantity=jacobian_quantity)

    # read in data
    freq = np.array(pyarts.xml.load("results/f_grid.xml"))
    jac = np.array(pyarts.xml.load("results/jacobian.xml"))
    alt = np.array(pyarts.xml.load("results/z_field.xml")).ravel()
    jac /= np.gradient(alt / 1000)  # normalize by layer thickness in km

    # plot jacobian
    highlight_frequency = 180e9  # Hz
    fig, ax = plt.subplots()
    freq_ind = argclosest(freq, highlight_frequency)
    plot_jacobian(alt, jac[freq_ind, :], jacobian_quantity=jacobian_quantity, ax=ax)

    plt.show()
