# %%
"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import pyarts.workspace
import numpy as np


def calc_olr_from_profiles(
    pressure_profile,
    temperature_profile,
    h2o_profile,
    N2=0.78,
    O2=0.21,
    CO2=400e-6,
    CH4=1.8e-6,
    O3=0.0,
    surface_altitude=0.0,
    nstreams=10,
    fnum=300,
    fmin=1e6,
    fmax=75e12,
    verbosity=0,
    version='2.6.8'
):
    """Calculate the outgoing-longwave radiation for a given atmosphere profiles.

    Parameters:

        pressure_profile (ndarray): Pressure profile [Pa].
        temperature_profile (ndarray): Temperature profile [K].
        h2o_profile (ndarray): Water vapor profile [VMR].
        N2 (float): Nitrogen volume mixing ratio. Defaults to 0.78.
        O2 (float): Oxygen volume mixing ratio. Defaults to 0.21.
        CO2 (float): Carbon dioxide volume mixing ratio. Defaults to 400 ppm.
        CH4 (float): Methane volume mixing ratio. Defaults to 1.8 ppm.
        O3 (float): Ozone volume mixing ratio. Defaults to 0.
        surface_altitude (float): Surface altitude [m]. Defaults to 0.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray: Frequency grid [Hz], OLR [Wm^-2]

    """
    
    pyarts.cat.download.retrieve(verbose=True, version=version)

    if fmin < 1e6:
        raise RuntimeError('fmin must be >= 1e6 Hz')

    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    ws.abs_speciesSet(
        species=[
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "CO2, CO2-CKDMT252",
            "CH4",
            "O2,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "O3",
        ]
    )

    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read cross section data
    ws.ReadXsecData(basename="lines/")

    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Calculate absorption
    ws.propmat_clearsky_agendaAuto()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)

    # Atmosphere and surface
    ws.Touch(ws.lat_grid)
    ws.Touch(ws.lon_grid)
    ws.lat_true = np.array([0.0])
    ws.lon_true = np.array([0.0])

    ws.AtmosphereSet1D()
    ws.p_grid = pressure_profile
    ws.t_field = temperature_profile[:, np.newaxis, np.newaxis]

    vmr_field = np.zeros((6, len(pressure_profile), 1, 1))
    vmr_field[0, :, 0, 0] = h2o_profile
    vmr_field[1, :, 0, 0] = CO2
    vmr_field[2, :, 0, 0] = CH4
    vmr_field[3, :, 0, 0] = O2
    vmr_field[4, :, 0, 0] = N2
    vmr_field[5, :, 0, 0] = O3
    ws.vmr_field = vmr_field

    ws.z_surface = np.array([[surface_altitude]])
    ws.p_hse = 100000
    ws.z_hse_accuracy = 100.0
    ws.z_field = 16e3 * (5 - np.log10(pressure_profile[:, np.newaxis, np.newaxis]))
    ws.atmfields_checkedCalc()
    ws.z_fieldFromHSE()

    # Set surface temperature equal to the lowest atmosphere level
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


# %%  Run module as script

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate example atmosphere
    # This atmosphere is not intended to be fully realistic, but to be simply
    # an example for the calculation of the OLR.

    # set pressure grid
    pressure_profile = np.linspace(1000e2, 1e2, 80)

    # create water vapor profile
    # Water vapor is simply define by a 1st order
    # polynomial in log-log space
    # log h2o = a + b * log pressure
    b = 4
    a = -6 - b * 4
    logH2O = a + b * np.log10(pressure_profile)
    H2O_profile = 10**logH2O

    # create temperature profile
    # Temperature is simply define by a 1st order
    # polynomial of log pressure
    # T = a + b * log pressure
    # For pressure < 100 hPa, the temperature is set to 200 K
    b = 100
    a = 200 - b * 4
    temperature_profile = a + b * np.log10(pressure_profile)
    temperature_profile[
        pressure_profile < 100e2
    ] = 200  # set temperature to 200 K below 100 hPa

    # plot atmosphere profiles
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(temperature_profile, pressure_profile / 100, label="Temperature")
    ax[0].set_xlabel("Temperature [K]")
    ax[0].set_ylabel("Pressure [hPa]")
    ax[0].invert_yaxis()

    ax[1].plot(H2O_profile, pressure_profile / 100, label="Water vapor")
    ax[1].set_xlabel("Water vapor [VMR]")
    ax[1].set_ylabel("Pressure [hPa]")
    ax[1].invert_yaxis()

    # %% calulate OLR from atmosphere profiles
    f_grid, olr = calc_olr_from_profiles(
        pressure_profile, temperature_profile, H2O_profile
    )

    # %% plot OLR
    fig, ax = plt.subplots()
    ax.plot(f_grid / 1e12, olr, label="Irradiance")
    ax.set_title(rf"OLR={np.trapz(olr, f_grid):3.2f} $\sf Wm^{{-2}}$")
    ax.set_xlim(f_grid.min() / 1e12, f_grid.max() / 1e12)
    ax.set_xlabel(r"Frequency [$\sf THz$]")
    ax.set_ylabel(r"Irradiance [$\sf Wm^{-2}Hz^{-1}$]")
    ax.set_ylim(bottom=0)

    plt.show()
