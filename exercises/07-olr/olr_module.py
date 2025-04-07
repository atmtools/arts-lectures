# %%
"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import pyarts.workspace
from pyarts.arts import constants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# %% Helper functions taken from the typhon package
def cmap2rgba(cmap=None, N=None, interpolate=True):
    """Convert a colormap into a list of RGBA values.

    Parameters:
        cmap (str): Name of a registered colormap.
        N (int): Number of RGBA-values to return.
            If ``None`` use the number of colors defined in the colormap.
        interpolate (bool): Toggle the interpolation of values in the
            colormap.  If ``False``, only values from the colormap are
            used. This may lead to the re-use of a color, if the colormap
            provides less colors than requested. If ``True``, a lookup table
            is used to interpolate colors (default is ``True``).

    Returns:
        ndarray: RGBA-values.

    Examples:
        >>> cmap2rgba('viridis', 5)
        array([[ 0.267004,  0.004874,  0.329415,  1.      ],
            [ 0.229739,  0.322361,  0.545706,  1.      ],
            [ 0.127568,  0.566949,  0.550556,  1.      ],
            [ 0.369214,  0.788888,  0.382914,  1.      ],
            [ 0.993248,  0.906157,  0.143936,  1.      ]])
    """
    cmap = plt.get_cmap(cmap)

    if N is None:
        N = cmap.N

    nlut = N if interpolate else None

    if interpolate and isinstance(cmap, colors.ListedColormap):
        # `ListedColormap` does not support lookup table interpolation.
        cmap = colors.LinearSegmentedColormap.from_list("", cmap.colors)
        return cmap(np.linspace(0, 1, N))

    return plt.get_cmap(cmap.name, lut=nlut)(np.linspace(0, 1, N))


def e_eq_water_mk(T):
    r"""Calculate the equilibrium vapor pressure of water over liquid water.

    .. math::
        \ln(e_\mathrm{liq}) &=
                    54.842763 - \frac{6763.22}{T} - 4.21 \cdot \ln(T) \\
                    &+ 0.000367 \cdot T
                    + \tanh \left(0.0415 \cdot (T - 218.8)\right) \\
                    &\cdot \left(53.878 - \frac{1331.22}{T}
                                 - 9.44523 \cdot \ln(T)
                                 + 0.014025 \cdot T \right)

    Parameters:
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Equilibrium vapor pressure [Pa].

    See also:
        :func:`~typhon.physics.e_eq_ice_mk`
            Calculate the equilibrium vapor pressure of water over ice.
        :func:`~typhon.physics.e_eq_mixed_mk`
            Calculate the vapor pressure of water over the mixed phase.

    References:
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
        ice and supercooled water for atmospheric applications,
        Quarterly Journal of the Royal Meteorological Society 131(608):
        1539–1565. doi:10.1256/qj.04.94

    """
    if np.any(T <= 0):
        raise ValueError("Temperatures must be larger than 0 Kelvin.")

    # Give the natural log of saturation vapor pressure over water in Pa

    e = (
        54.842763
        - 6763.22 / T
        - 4.21 * np.log(T)
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8))
        * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)
    )

    return np.exp(e)


def relative_humidity2vmr(RH, p, T, e_eq=None):
    r"""Convert relative humidity into water vapor VMR.

    .. math::
        x = \frac{\mathrm{RH} \cdot e_s(T)}{p}

    Note:
        By default, the relative humidity is calculated with respect to
        saturation over liquid water in accordance to the WMO standard for
        radiosonde observations.
        You can use :func:`~typhon.physics.e_eq_mixed_mk` to calculate
        relative humidity with respect to saturation over the mixed-phase
        following the IFS model documentation.

    Parameters:
        RH (float or ndarray): Relative humidity.
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Volume mixing ratio [unitless].

    See also:
        :func:`~typhon.physics.vmr2relative_humidity`
            Complement function (returns RH for given VMR).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> relative_humidity2vmr(0.75, 101300, 300)
        0.026185323887350429
    """
    if e_eq is None:
        e_eq = e_eq_water_mk

    return RH * e_eq(T) / p


def vmr2relative_humidity(vmr, p, T, e_eq=None):
    r"""Convert water vapor VMR into relative humidity.

    .. math::
        \mathrm{RH} = \frac{x \cdot p}{e_s(T)}

    Note:
        By default, the relative humidity is calculated with respect to
        saturation over liquid water in accordance to the WMO standard for
        radiosonde observations.
        You can use :func:`~typhon.physics.e_eq_mixed_mk` to calculate
        relative humidity with respect to saturation over the mixed-phase
        following the IFS model documentation.

    Parameters:
        vmr (float or ndarray): Volume mixing ratio,
        p (float or ndarray): Pressure [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Relative humidity [unitless].

    See also:
        :func:`~typhon.physics.relative_humidity2vmr`
            Complement function (returns VMR for given RH).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> vmr2relative_humidity(0.025, 1013e2, 300)
        0.71604995533615401
    """
    if e_eq is None:
        e_eq = e_eq_water_mk

    return vmr * p / e_eq(T)


def planck(f, T):
    """Calculate black body radiation for given frequency and temperature.

    Parameters:
        f (float or ndarray): Frquencies [Hz].
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Radiances.

    """
    c = constants.c
    h = constants.h
    k = constants.k

    return 2 * h * f**3 / (c**2 * (np.exp(np.divide(h * f, (k * T))) - 1))


# %% main functions
def Change_T_with_RH_const(atmfield, DeltaT=0.0):
    """Change the temperature everywhere in the atmosphere by a value of DeltaT
       but without changing the relative humidity. This results in a changed
       volume mixing ratio of water vapor.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        DeltaT (float): Temperature change [K].

    Returns:
        GriddedField4: Atmosphere field
    """

    # water vapor
    vmr = atmfield.get("abs_species-H2O")

    # Temperature
    T = atmfield.get("T")

    # Reshape pressure p, so that p has the same dimensions
    p = atmfield.grids[1][:].reshape(T.shape)

    # Calculate relative humidity
    rh = vmr2relative_humidity(vmr, p, T)

    # Calculate water vapor volume mixing ratio for changed temperature
    vmr = relative_humidity2vmr(rh, p, T + DeltaT)

    # update atmosphere field
    atmfield.set("T", T + DeltaT)
    atmfield.set("abs_species-H2O", vmr)

    return atmfield


def calc_olr_from_atmfield(
    atmfield,
    nstreams=10,
    fnum=300,
    fmin=1.0,
    fmax=75e12,
    species="default",
    verbosity=0,
):
    """Calculate the outgoing-longwave radiation for a given atmosphere.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        species (List of strings): List fo absorption species. Defaults to "default"
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray: Frequency grid [Hz], OLR [Wm^-2]
    """

    pyarts.cat.download.retrieve(verbose=bool(verbosity))

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Definition of species
    if species == "default":
        ws.abs_speciesSet(
            species=[
                "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
                "CO2, CO2-CKDMT252",
            ]
        )
    else:
        ws.abs_speciesSet(species=species)

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read cross section data
    ws.ReadXsecData(basename="lines/")

    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Calculate absorption
    ws.propmat_clearsky_agendaAuto()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    # Atmosphere and surface
    ws.atm_fields_compact = atmfield
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # Set surface height and temperature equal to the lowest atmosphere level
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.surface_skin_t = ws.t_field.value[0, 0, 0]

    # Output radiance not converted
    ws.StringSet(ws.iy_unit, "1")

    # set cloudbox to full atmosphere
    ws.cloudboxSetFullAtm()

    # set particle scattering to zero, because we want only clear sky
    ws.scat_data_checked = 1
    ws.Touch(ws.scat_data)
    ws.pnd_fieldZero()

    # No sensor properties
    ws.sensorOff()

    # No jacobian calculations
    ws.jacobianOff()

    # Check model atmosphere
    ws.scat_data_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.lbl_checkedCalc()

    # Perform RT calculations
    ws.spectral_irradiance_fieldDisort(nstreams=nstreams, emission=1)

    olr = ws.spectral_irradiance_field.value[:, -1, 0, 0, 1][:]

    return ws.f_grid.value[:], olr
