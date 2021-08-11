"""Calculate and plot clear-sky Jacobians."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pyarts
import typhon as ty

from bokeh.plotting import figure
from bokeh.models import Span, Label, BasicTickFormatter
from pyarts import xml

ASPECT_RATIO = 3 / 2
SUBPLOT_WIDTH = 400


def argclosest(array, value):
    """Returns the index in ``array`` which is closest to ``value``."""
    return np.abs(array - value).argmin()


def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


def plot_brightness_temperature(
    frequency, y, where=None, plotsize=SUBPLOT_WIDTH, aspect=ASPECT_RATIO
):
    p = figure(
        x_range=(frequency.min() / 1e9, frequency.max() / 1e9),
        tooltips=[("x", "$x"), ("value", "@y")],
        width=plotsize,
        aspect_ratio=aspect,
    )

    p.line(frequency / 1e9, y, line_width=2)
    p.xaxis.axis_label = "Frequency / GHz"
    p.yaxis.axis_label = "T_B / K"
    p.xgrid.visible = p.ygrid.visible = False
    p.xaxis.minor_tick_line_color = p.yaxis.minor_tick_line_color = None

    if where is not None:
        freq_ind = argclosest(frequency, where)
        p.circle(
            x=[frequency[freq_ind] / 1e9],
            y=[y[freq_ind]],
            size=10,
            color=["red"],
            fill_alpha=0.8,
        )
        label = Label(
            x=frequency[freq_ind] / 1e9,
            y=y[freq_ind],
            y_offset=3,
            text=f"{frequency[freq_ind]/1e9:.2f} GHz",
            text_align="center",
            render_mode="css",
            text_color="red",
        )
        p.add_layout(label)

    return p


def plot_opacity(
    frequency, opacity, where=None, plotsize=SUBPLOT_WIDTH, aspect=ASPECT_RATIO
):
    p = figure(
        y_axis_type="log",
        x_range=(frequency.min() / 1e9, frequency.max() / 1e9),
        y_range=(0.4, 70),
        tooltips=[("x", "$x"), ("value", "@y")],
        width=plotsize,
        aspect_ratio=aspect,
    )

    p.line(frequency / 1e9, opacity[-1, :], line_width=2)
    hline = Span(location=1, dimension="width", line_color="gray", line_width=0.8)
    p.renderers.extend([hline])

    p.xaxis.axis_label = "Frequency / GHz"
    p.yaxis.axis_label = "Zenith Opacity"
    p.xgrid.visible = p.ygrid.visible = False
    p.xaxis.minor_tick_line_color = p.yaxis.minor_tick_line_color = None

    if where is not None:
        freq_ind = argclosest(frequency, where)
        p.circle(
            x=[frequency[freq_ind] / 1e9],
            y=[opacity[-1, freq_ind]],
            size=10,
            color=["red"],
            fill_alpha=0.8,
        )

    return p


def plot_jacobian(
    height, jacobian, jacobian_quantity, plotsize=SUBPLOT_WIDTH, aspect=ASPECT_RATIO):
    p = figure(
        y_axis_type="linear",
        y_range=(0.4, 20),
        tooltips=[("x", "$x"), ("value", "@y")],
        width=plotsize,
        aspect_ratio=aspect,
    )

    l = p.line(jacobian, height / 1000.0, line_width=2)

    unit = "K/K/km" if jacobian_quantity == "T" else "K/1/km"
    p.xaxis.axis_label = f"{jacobian_quantity} Jacobian / {unit}"
    p.yaxis.axis_label = "z / km"
    p.xgrid.visible = p.ygrid.visible = False
    p.xaxis.minor_tick_line_color = p.yaxis.minor_tick_line_color = None

    jac_peak = height[np.abs(jacobian).argmax()] / 1000.0
    hline = Span(
        location=jac_peak, dimension="width", line_color="black", line_width=1.5
    )
    p.renderers.extend([hline])

    label = Label(
        x=5,
        y=jac_peak,
        x_units="screen",
        text=f"{jac_peak:.2f} km",
        text_color="black",
        render_mode="css",
    )
    p.add_layout(label)

    return p


def plot_opacity_profile(height, opacity, plotsize=SUBPLOT_WIDTH, aspect=ASPECT_RATIO):
    p = figure(
        x_axis_type="log",
        y_axis_type="linear",
        x_range=(1e-8, 1e2),
        y_range=(0.4, 20),
        tooltips=[("x", "$x"), ("value", "@y")],
        width=plotsize,
        aspect_ratio=aspect,
    )

    p.line(opacity, height[::-1] / 1000.0, line_width=2)

    p.xaxis.ticker = 10.0 ** np.arange(-8, 4, 2)
    p.xaxis.axis_label = r"Opacity 𝜏(z, z_TOA)"
    p.yaxis.axis_label = "z / km"
    p.xgrid.visible = p.ygrid.visible = False
    p.xaxis.minor_tick_line_color = p.yaxis.minor_tick_line_color = None

    try:
        tau1 = height[::-1][np.where(opacity >= 1)[0][0]]
    except IndexError:
        pass
    else:
        tau1 /= 1000
        hline = Span(
            location=tau1, dimension="width", line_color="black", line_width=1.5
        )
        p.renderers.extend([hline])
        label = Label(
            x=5,
            y=tau1,
            x_units="screen",
            text=f"{tau1:.2f} km",
            text_color="black",
            render_mode="css",
        )
        p.add_layout(label)

        vline = Span(location=1, dimension="height", line_color="grey", line_width=0.8)
        p.renderers.extend([vline])

    return p


def calc_jacobians(
    jacobian_quantity="H2O", fmin=150e9, fmax=200e9, fnum=200, verbosity=0
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
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="spectroscopy/Hitran/")

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
