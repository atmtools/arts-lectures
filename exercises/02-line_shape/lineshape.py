# %% Import modules and define functions
"""Calculate and plot absorption cross sections."""
import re

import numpy as np
import pyarts
import scipy as sp


def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


def linewidth(f, a):
    """Calculate the full-width at half maximum (FWHM) of an absorption line.

    Parameters:
        f (ndarray): Frequency grid.
        a (ndarray): Line properties
            (e.g. absorption coefficients or cross-sections).

    Returns:
        float: Linewidth.

    Examples:
        >>> f = np.linspace(0, np.pi, 100)
        >>> a = np.sin(f)**2
        >>> linewidth(f, a)
        1.571048056449009
    """

    idx = np.argmax(a)

    if idx < 3 or idx > len(a) - 3:
        raise RuntimeError(
            "Maximum is located too near at the edge.\n"
            + "Could not found any peak. \n"
            + "Please adjust the frequency range."
        )

    s = sp.interpolate.UnivariateSpline(f, a - np.max(a) / 2, s=0)

    zeros = s.roots()
    sidx = np.argsort((zeros - f[idx]) ** 2)

    if zeros.size == 2:
        logic = zeros[sidx] > f[idx]

        if np.sum(logic) == 1:
            fwhm = abs(np.diff(zeros[sidx])[0])

        else:
            print(
                "I only found one half maxima.\n"
                + "You should adjust the frequency range to have more reliable results.\n"
            )

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2

    elif zeros.size == 1:
        fwhm = abs(zeros[0] - f[idx]) * 2

        print(
            "I only found one half maxima.\n"
            + "You should adjust the frequency range to have more reliable results.\n"
        )

    elif zeros.size > 2:
        sidx = sidx[0:2]

        logic = zeros[sidx] > f[idx]

        print(
            "It seems, that there are more than one peak"
            + " within the frequency range.\n"
            + "I stick to the maximum peak.\n"
            + "But I would suggest to adjust the frequevncy range. \n"
        )

        if np.sum(logic) == 1:
            fwhm = abs(np.diff(zeros[sidx])[0])

        else:
            print(
                "I only found one half maxima.\n"
                + "You should adjust the frequency range to have more reliable results.\n"
            )

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2

    elif zeros.size == 0:
        raise RuntimeError(
            "Could not found any peak. :( \n"
            + "Probably, frequency range is too small.\n"
        )

    return fwhm


def calculate_absxsec(
    species="N2O",
    pressure=800e2,
    temperature=300.0,
    fmin=10e9,
    fmax=2000e9,
    fnum=10_000,
    lineshape="VP",
    normalization="VVH",
    verbosity=0,
    ws=None,
    vmr=0.05,
    lines_off=0
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
                            Available options:
                            DP        -      Doppler profile,
                            LP        -      Lorentz profile,
                            VP        -      Voigt profile,
                            SDVP      -      Speed-dependent Voigt profile,
                            HTP       -      Hartman-Tran profile.
        normalization (str): Line shape normalization factor.
                            Available options:
                            VVH       -      Van Vleck and Huber,
                            VVW       -      Van Vleck and Weisskopf,
                            RQ        -      Rosenkranz quadratic,
                            None      -      No extra normalization.
        verbosity (int): Set ARTS verbosity (``0`` prevents all output).
        vmr (float): Volume mixing ratio. This is mainly important for the
                     water vapor continua.
        lines_off (int): Switch off lines, if no contnua is selected,
                         absorption will be zero.

    Returns:
        ndarray, ndarray: Frequency grid [Hz], Abs. cross sections [m^2]
    """
    # Create ARTS workspace and load default settings
    reload = False
    pyarts.cat.download.retrieve(verbose=bool(verbosity))

    if ws is not None:
        # check if species fits to cached species
        temp = str(ws.abs_species.value[0][0])
        species_cache = temp.split("-")[0]

        if species != species_cache:
            print(
                f"Cached species:{species_cache} \n"
                f"Desired species:{species} \n"
                "As the chached and the desired species are different,\n"
                "I have to read in the catalog..."
            )
            reload = True

    if ws is None or reload:
        ws = pyarts.workspace.Workspace(verbosity=0)
        ws.water_p_eq_agendaSet()
        ws.PlanetSet(option="Earth")

        ws.verbositySetScreen(ws.verbosity, verbosity)

        # We do not want to calculate the Jacobian Matrix
        ws.jacobianOff()

        # Define absorption species
        ws.abs_speciesSet(species=[species])
        ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesNormalization(option=normalization)
    ws.abs_lines_per_speciesTurnOffLineMixing()
    if lines_off:
        ws.abs_lines_per_speciesSetEmpty()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, fnum, fmin, fmax)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric settings
    ws.AtmosphereSet1D()
    ws.stokes_dim = 1

    # Setting the pressure, temperature and vmr
    ws.rtp_pressure = float(pressure)  # [Pa]
    ws.rtp_temperature = float(temperature)  # [K]
    ws.rtp_vmr = np.array([vmr])  # [VMR]
    ws.Touch(ws.rtp_nlte)

    # isotop
    ws.isotopologue_ratiosInitFromBuiltin()

    # Calculate absorption cross sections
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agenda_checked = 1
    ws.propmat_clearskyInit()
    ws.propmat_clearskyAddLines()
    ws.propmat_clearskyAddPredefined()

    # Convert abs coeff to cross sections on return
    number_density = pressure * vmr / (pyarts.arts.constants.k * temperature)

    return (
        ws.f_grid.value.value.copy(),
        ws.propmat_clearsky.value.data.value[0, 0, :, 0].copy() / number_density,
        ws,
    )


# %%  Run module as script
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    species = "H2O"
    temperature = 300  # [K]
    pressure = 101325  # [Pa]
    cache = None  # cache for ARTS workspace

    # Call ARTS to calculate absorption cross sections
    freq, abs_xsec, cache = calculate_absxsec(species, pressure, temperature, fnum=1000, ws=cache)

    fig, ax = plt.subplots()
    ax.plot(freq, abs_xsec)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(r"Abs. cross section [$\sf m^2$]")
    plt.show()

    # recall ARTS to calculate absorption cross sections at another pressure
    species = "H2O"
    temperature = 300  # [K]
    pressure = 10132.5  # [Pa]

    freq, abs_xsec, cache = calculate_absxsec(species, pressure, temperature, fnum=1000, ws=cache)

    fig, ax = plt.subplots()
    ax.plot(freq, abs_xsec)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(r"Abs. cross section [$\sf m^2$]")
    plt.show()
    
    # recall ARTS to calculate absorption cross sections for another species
    species = "O3"
    temperature = 300  # [K]
    pressure = 10132.5  # [Pa]

    freq, abs_xsec, cache = calculate_absxsec(species, pressure, temperature, fnum=1000, ws=cache)

    fig, ax = plt.subplots()
    ax.plot(freq, abs_xsec)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(r"Abs. cross section [$\sf m^2$]")
    plt.show()
    

