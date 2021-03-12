"""Calculate and plot absorption cross sections."""
import re

import numpy as np
import scipy as sp
import pyarts


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
        raise RuntimeError('Maximum is located too near at the edge.\n' +
                           'Could not found any peak. \n' +
                           'Please adjust the frequency range.')

    s = sp.interpolate.UnivariateSpline(f, a - np.max(a) / 2, s=0)

    zeros = s.roots()
    sidx = np.argsort((zeros - f[idx]) ** 2)

    if zeros.size == 2:

        logic = zeros[sidx] > f[idx]

        if np.sum(logic) == 1:

            fwhm = abs(np.diff(zeros[sidx])[0])

        else:

            print('I only found one half maxima.\n'
                  + 'You should adjust the frequency range to have more reliable results.\n')

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2


    elif zeros.size == 1:

        fwhm = abs(zeros[0] - f[idx]) * 2

        print('I only found one half maxima.\n'
              + 'You should adjust the frequency range to have more reliable results.\n')

    elif zeros.size > 2:

        sidx = sidx[0:2]

        logic = zeros[sidx] > f[idx]

        print('It seems, that there are more than one peak'
              + ' within the frequency range.\n'
              + 'I stick to the maximum peak.\n'
              + 'But I would suggest to adjust the frequevncy range. \n')

        if np.sum(logic) == 1:

            fwhm = abs(np.diff(zeros[sidx])[0])

        else:

            print('I only found one half maxima.\n'
                  + 'You should adjust the frequency range to have more reliable results.\n')

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2



    elif zeros.size == 0:

        raise RuntimeError('Could not found any peak. :( \n' +
                           'Probably, frequency range is too small.\n')

    return fwhm

def calculate_absxsec(
    species="N2O",
    pressure=800e2,
    temperature=300.0,
    fmin=10e9,
    fmax=2000e9,
    fnum=10_000,
    lineshape="LP",
    normalization="RQ",
    ws=None,
    verbosity=0
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
        ws (workspace): Cached ARTS workspace.  If set to ``None`` caching
                                is not used.
        verbosity (int): Set ARTS verbosity (``0`` prevents all output).

    Returns:
        ndarray, ndarray, workspace: Frequency grid [Hz], Abs. cross sections [m^2],  ARTS workspace
    """
    # Create ARTS workspace and load default settings
    reload = False

    if ws is not None:
        # check if species fits to cached species
        species_cache = ws.abs_species.value[0][0].split('-')[0]

        if species == species_cache:
            ws.Copy(ws.abs_species, ws.abs_species_cache)
            ws.Copy(ws.abs_lines_per_species, ws.abs_lines_per_species_cache)

        else:
            print(f'Cached species:{species_cache} \n'
                  f'Desired species:{species} \n'
                  'As the chached and the desired species are different,\n'
                  'I have to read in the catalog...')
            reload = True

    if ws is None or reload:
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
           basename="spectroscopy/Artscat/"
        )

        ws.ArrayOfArrayOfSpeciesTagCreate('abs_species_cache')
        ws.ArrayOfArrayOfAbsorptionLinesCreate('abs_lines_per_species_cache')

        ws.Copy(ws.abs_species_cache, ws.abs_species)
        ws.Copy(ws.abs_lines_per_species_cache, ws.abs_lines_per_species)

    ws.abs_lines_per_speciesSetLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesSetCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesSetNormalization(option=normalization)

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, fnum, fmin, fmax)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric settings
    ws.AtmosphereSet1D()

    # Setting the pressure, temperature and vmr
    ws.NumericSet(ws.rtp_pressure, float(pressure))  # [Pa]
    ws.NumericSet(ws.rtp_temperature, float(temperature))  # [K]
    ws.VectorSet(ws.rtp_vmr, np.array([1.0]))  # [VMR]
    ws.Touch(ws.abs_nlte)

    ws.AbsInputFromRteScalars()

    # isotop
    ws.isotopologue_ratiosInitFromBuiltin()

    # Calculate absorption cross sections
    ws.lbl_checkedCalc()
    ws.abs_xsec_agenda_checkedCalc()
    ws.abs_xsec_per_speciesInit()
    ws.abs_xsec_per_speciesAddLines()

    return ws.f_grid.value.copy(), ws.abs_xsec_per_species.value.copy()[0], ws
