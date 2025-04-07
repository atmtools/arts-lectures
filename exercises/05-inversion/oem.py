"""Perform an OEM retrieval and plot the results."""
import numpy as np
import pyarts.workspace
from pyarts import xml


def forward_model(f_grid, atm_fields_compact, retrieval_quantity='H2O',verbosity=0):
    """Perform a radiative transfer simulation.

    Parameters:
        f_grid (ndarray): Frequency grid [Hz].
        atm_fields_compact (GriddedField4): Atmosphere field.
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray: Frequency grid [Hz], Jacobian [K/1]
    """

    pyarts.cat.download.retrieve(verbose=bool(verbosity))

    ws = pyarts.workspace.Workspace(verbosity=verbosity)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # standard emission agenda
    ws.iy_main_agendaSet(option="Emission")

    # cosmic background radiation
    ws.iy_space_agendaSet(option="CosmicBackground")

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    # sensor-only path
    ws.ppath_agendaSet(option="FollowSensorLosPath")

    # no refraction
    ws.ppath_step_agendaSet(option="GeometricPath")

    # Non reflecting surface
    ws.surface_rtprop_agendaSet(option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    #########################################################################

    # Definition of absorption species
    ws.abs_speciesSet(
        species=[
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "O2-TRE05",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        ]
    )

    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesNormalization(option=normalization)

    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.4)

    # Set the frequency grid
    ws.f_grid = f_grid

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # No sensor properties
    ws.sensorOff()

    # We select here to use Planck brightness temperatures
    ws.StringSet(ws.iy_unit, "PlanckBT")

    #########################################################################

    # Atmosphere and surface
    ws.AtmosphereSet1D()
    ws.atm_fields_compact = atm_fields_compact
    ws.atm_fields_compactAddConstant(
        ws.atm_fields_compact, "abs_species-N2", 0.78, 0, ["abs_species-H2O"]
    )
    ws.atm_fields_compactAddConstant(
        ws.atm_fields_compact, "abs_species-O2", 0.21, 0, ["abs_species-H2O"]
    )
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    # Definition of sensor position and line of sight (LOS)
    ws.MatrixSet(ws.sensor_pos, np.array([[10e3]]))
    ws.MatrixSet(ws.sensor_los, np.array([[0]]))
    ws.sensorOff()

    # Jacobian calculation
    ws.jacobianInit()
    if retrieval_quantity=='H2O':
        ws.jacobianAddAbsSpecies(
            g1=ws.p_grid,
            g2=ws.lat_grid,
            g3=ws.lon_grid,
            species="H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            unit="vmr",
        )
    elif retrieval_quantity=='Temperature':
        ws.jacobianAddTemperature(
            g1=ws.p_grid,
            g2=ws.lat_grid,
            g3=ws.lon_grid
            )
    else:
        raise ValueError('only H2O or Temperature are allowed as retrieval quantity')
    ws.jacobianClose()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # on-the-fly absorption
    ws.propmat_clearsky_agendaAuto()

    # Perform RT calculations
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.yCalc()

    return ws.y.value[:].copy(), ws.jacobian.value[:].copy()

# %%
if __name__ == "__main__":

    import matplotlib.pyplot as plt


    # Load the (simulated) measurement.
    measurement = xml.load("input/measurement.xml")
    f_grid = measurement.grids[0]
    y_measurement = measurement.data

    # Load the a priori atmospheric state.
    atm_fields = xml.load("input/x_apriori.xml")
    z = atm_fields.get("z", keep_dims=False)
    x_apriori = atm_fields.get("abs_species-H2O", keep_dims=False)

    # Load the covariance matrices.
    S_xa = xml.load("input/S_xa.xml")
    S_y = 2.5e-3 * np.eye(f_grid.size)  # in [K^2]


    y, K = forward_model(f_grid, atm_fields)


    # Plot absorption cross sections
    fig, ax = plt.subplots()
    ax.plot(f_grid, y)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Brightness Temperature [$\sf K$]")
    plt.show()
    fig.savefig('forward_spectra.pdf')