import numpy as np
from ..leja19_smf import leja19_continuity_model_smf

from ..leja19_smf import (
    DEFAULT_DBL_SCHECHTER,
    DEFAULT_DBL_SCHECHTER_FITTER,
    DBL_SCHECHTER_FITTER_BOUNDS,
    _double_schechter,
    _double_schechter_fitter,
)


LOGSM_Z1 = np.array(
    (
        8.044,
        8.169,
        8.293,
        8.422,
        8.551,
        8.663,
        8.771,
        8.862,
        8.953,
        9.045,
        9.136,
        9.227,
        9.319,
        9.410,
        9.501,
        9.593,
        9.684,
        9.775,
        9.867,
        9.958,
        10.049,
        10.141,
        10.232,
        10.323,
        10.414,
        10.506,
        10.597,
        10.688,
        10.780,
        10.871,
        10.962,
        11.050,
        11.129,
        11.195,
        11.249,
        11.295,
        11.337,
        11.378,
        11.416,
        11.449,
        11.483,
        11.512,
        11.537,
        11.562,
        11.587,
        11.612,
        11.633,
        11.657,
        11.671,
        11.685,
    )
)
LOGSM_Z2 = np.array(
    (
        8.558,
        8.687,
        8.812,
        8.941,
        9.051,
        9.143,
        9.211,
        9.320,
        9.452,
        9.556,
        9.619,
        9.735,
        9.827,
        9.918,
        10.010,
        10.101,
        10.193,
        10.284,
        10.370,
        10.467,
        10.570,
        10.646,
        10.950,
        11.044,
        11.129,
        11.207,
        11.270,
        11.324,
        11.373,
        11.415,
        11.452,
        11.485,
        11.518,
        11.552,
        11.580,
        11.605,
        11.630,
        11.647,
    )
)
LOGPHI_Z1 = np.array(
    (
        -1.258,
        -1.318,
        -1.378,
        -1.440,
        -1.502,
        -1.557,
        -1.607,
        -1.649,
        -1.693,
        -1.736,
        -1.779,
        -1.821,
        -1.863,
        -1.906,
        -1.946,
        -1.986,
        -2.024,
        -2.061,
        -2.097,
        -2.131,
        -2.163,
        -2.193,
        -2.223,
        -2.250,
        -2.276,
        -2.304,
        -2.336,
        -2.372,
        -2.417,
        -2.477,
        -2.553,
        -2.649,
        -2.761,
        -2.875,
        -2.986,
        -3.095,
        -3.205,
        -3.331,
        -3.452,
        -3.576,
        -3.707,
        -3.832,
        -3.950,
        -4.072,
        -4.204,
        -4.344,
        -4.493,
        -4.631,
        -4.734,
        -4.836,
    )
)
LOGPHI_Z2 = np.array(
    (
        -1.838,
        -1.910,
        -1.957,
        -2.016,
        -2.076,
        -2.120,
        -2.154,
        -2.214,
        -2.266,
        -2.308,
        -2.339,
        -2.405,
        -2.448,
        -2.495,
        -2.535,
        -2.569,
        -2.615,
        -2.660,
        -2.691,
        -2.749,
        -2.804,
        -2.831,
        -3.030,
        -3.137,
        -3.231,
        -3.337,
        -3.456,
        -3.561,
        -3.668,
        -3.772,
        -3.892,
        -4.015,
        -4.118,
        -4.248,
        -4.359,
        -4.493,
        -4.622,
        -4.701,
    )
)


def test_leja19_smf_agrees_with_published_plot():
    """Webplotdigitizer comparison against dark blue and light orange curves
    appearing in the right-hand panel of Figure 6, arXiv:1910.04168.
    """
    implemented_logphi1 = np.log10(leja19_continuity_model_smf(LOGSM_Z1, 0.3))
    assert np.allclose(LOGPHI_Z1, implemented_logphi1, atol=0.05)

    implemented_logphi2 = np.log10(leja19_continuity_model_smf(LOGSM_Z2, 1.7))
    assert np.allclose(LOGPHI_Z2, implemented_logphi2, atol=0.05)


def test_leja19_smf_accepts_scalars_or_ndarrays():
    """leja19_continuity_model_smf should accept a float or ndarray
    for either log10m or z arguments.
    """
    phi1 = leja19_continuity_model_smf(LOGSM_Z1, 0)
    phi2 = leja19_continuity_model_smf(LOGSM_Z1, np.zeros_like(LOGSM_Z1))
    assert phi1.shape == phi2.shape
    assert np.allclose(phi1, phi2)

    zarr = np.linspace(0, 10, 1000)
    phi3 = leja19_continuity_model_smf(10, zarr)
    phi4 = leja19_continuity_model_smf(np.zeros_like(zarr) + 10, zarr)
    assert phi3.shape == phi4.shape
    assert np.allclose(phi3, phi4)


def test_fitter_dict_agrees_with_default():
    orig, fitter = DEFAULT_DBL_SCHECHTER, DEFAULT_DBL_SCHECHTER_FITTER
    assert np.allclose(orig["logphi1"], fitter["logphi1"])
    assert np.allclose(orig["logphi2"], fitter["logphi1"] + fitter["delta_logphi1"])
    assert np.allclose(orig["log10mstar"], fitter["log10mstar"])
    assert np.allclose(orig["alpha1"], fitter["alpha1"])
    assert np.allclose(orig["alpha2"], fitter["alpha1"] + fitter["delta_alpha1"])


def test_fitter_dict_params_are_within_bounds():
    for key, bound in DBL_SCHECHTER_FITTER_BOUNDS.items():
        lo, hi = bound
        assert lo < DEFAULT_DBL_SCHECHTER_FITTER[key] < hi


def test_dbl_schechter_fitter():
    p0 = np.array(list(DEFAULT_DBL_SCHECHTER.values()))
    p1 = np.array(list(DEFAULT_DBL_SCHECHTER_FITTER.values()))

    logsm = np.linspace(8, 12, 500)
    smf1 = _double_schechter(logsm, *p0)
    smf2 = _double_schechter_fitter(logsm, *p1)
    assert np.allclose(smf1, smf2, rtol=0.01)
