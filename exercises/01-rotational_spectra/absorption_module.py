"""Calculate and plot absorption cross sections."""
import re

import matplotlib.pyplot as plt
import numpy as np
import typhon as ty
import pyarts
from typhon.plots import styles


def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


def calculate_absxsec(
    species="N2O",
    pressure=800e2,
    temperature=300.0,
    fmin=10e9,
    fmax=2000e9,
    fnum=10_000,
    lineshape="LP",
    normalization="RQ",
    verbosity=2
):
    """Calculate absorption cross sections.

    Parameters:
        species (str): Absorption species name.
        pressure (float): Atmospheric pressure [Pa].
        temperature (float): Atmospheric temperature [K].
        fmin (float): Minimum frequency [Hz].
        fmax (float): Maximum frequency [Hz].
        fnum (int): Number of frequency grid points.
        lineshape (str): Line shape model.
        normalization (str): Line shape normalization factor.
        verbosity (int): Set ARTS verbosity (``0`` prevents all output).

    Returns:
        ndarray, ndarray: Frequency grid [Hz], Abs. cross sections [m^2]
    """
    # Create ARTS workspace and load default settings
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # We do not want to calculate the Jacobian Matrix
    ws.jacobianOff()

    # Agenda for scalar gas absorption calculation
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    # Define absorption species
    ws.abs_speciesSet(species=[species])
    ws.ArrayOfIndexSet(ws.abs_species_active, [0])

    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
       basename="spectroscopy/Hitran/"
    )

    ws.abs_lines_per_speciesSetLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesSetCutoff(option="None", value=0.0)
    ws.abs_lines_per_speciesSetNormalization(option=normalization)

    # Atmospheric settings
    ws.AtmosphereSet1D()

    # Setting the pressure, temperature and vmr
    ws.NumericSet(ws.rtp_pressure, float(pressure))  # [Pa]
    ws.NumericSet(ws.rtp_temperature, float(temperature))  # [K]
    ws.VectorSet(ws.rtp_vmr, np.array([1.0]))  # [VMR]
    ws.Touch(ws.abs_nlte)

    ws.AbsInputFromRteScalars()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, fnum, fmin, fmax)

    # isotop
    ws.isotopologue_ratiosInitFromBuiltin()

    # Calculate absorption cross sections
    ws.lbl_checkedCalc()
    ws.abs_xsec_agenda_checkedCalc()
    ws.abs_xsec_per_speciesInit()
    ws.abs_xsec_per_speciesAddLines()

    return ws.f_grid.value.copy(), ws.abs_xsec_per_species.value.copy()[0]
