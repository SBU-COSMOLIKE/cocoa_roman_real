"""Project choices for the shared galaxy/shear covariance calculation.

The covariance notebook (EXAMPLE_EVALUATE_COVARIANCE.ipynb) and the
command-line runner compute_covariance.py share this module, the survey
adapter: configuration() returns every survey, cosmology and accuracy
choice, initialize() installs them in the compiled interface, and compute()
returns the matrices. The numerical algorithms live in
cosmolike_notebook_utils.covariance.

The example uses the project's 8 lens and 8 source distributions. It is a
massless-neutrino forecast with explicit Gaussian non-Limber and IA choices
and number densities; it does not reproduce the project's frozen
likelihood.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    The covariance README records the catalog assumptions and their sources.
    Redshift-file normalization sets a shape, not a catalog number density.

    Arguments:
        accuracy_boost = None uses default.yaml; 1, 2, 4 or 8 refines it.
        gaussian = optional nonlimber/ia/A1/A2/B_TA model mapping.
        accuracy_overrides = named internal controls from default.yaml;
            **accuracy_overrides collects every other keyword argument
            into this dict.
    Returns:
        Fully resolved settings, including the unboosted accuracy parameters.
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # 16 rounded log-spaced edges give 15 Fourier bands from l = 30 to 4000.
    # Band i covers the integers band_first[i] .. band_last[i], both
    # included, so every integer multipole is counted once.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once.
    # mnu = 0: the forecast uses massless neutrinos.
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.04,
            "H0": 67.32,
            "ns": 0.96605,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # n(z) files: each column gives the shape of one bin's redshift
        # distribution. The two flags mirror the likelihood yaml:
        # interpolation 0 = cubic spline, z column 0 = left bin edges.
        "lens_file": "data/example1.nz",
        "source_file": "data/example1.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # Layout choices kept fixed when the accuracy is refined: the
        # [lens, source] pairs absent from gamma_t (zero-based), the Fourier
        # bands and the panels of the halo-mass integrals (ln M edges).
        "excluded_gammat": [[6, 0], [7, 0], [7, 1]],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # Forecast assumptions (covariance/README.md, "Which survey does
        # the example use?"): 2415 square degrees, 41.3 galaxies per square
        # arcminute in each catalog split equally into 8 bins, shape noise
        # 0.30 per ellipticity component, and the project's linear galaxy
        # bias of each lens bin.
        "area_deg2": 2415.0,
        "lens_density_arcmin2": [41.3/8]*8,
        "source_density_arcmin2": [41.3/8]*8,
        "sigma_e_component": [0.30]*8,
        "bias": [1.18, 1.40, 1.55, 1.71, 1.90, 2.15, 2.52, 2.44],

        # 16 log-spaced edges: the 15 angular bins of the dataset, 2.5 to
        # 250 arcmin. a_edges: scale-factor panels of the radial quadrature,
        # increasing in a from z = 4 to the observer end, z = 1e-5.
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=16),
        "a_edges": 1.0/(1.0+np.array([4.0, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_roman_real_interface module.
        settings = resolved mapping from configuration().

    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    Arguments:
        interface = the compiled project module, after initialize().
        settings = the mapping from configuration().
        space = "real" or "fourier".
        rows = optional subset of measured rows.
        progress = optional callback, called as progress(stage,
            elapsed_seconds).
        backend = None for the notebook wrappers, interface.covariance for
            the command-line runner.

    Returns:
        the shared forecast dict, including the resolved settings and the
        coordinates. The full real-space layout has 2115 entries; the
        Fourier layout has 1575.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
