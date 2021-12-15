"""Stellar mass function taken from Leja+19 continuity model."""
import numpy as np
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp


__all__ = ("leja19_continuity_model_smf",)

ZOBS = 0.2
DEFAULT_PARAMS = OrderedDict(
    logphi1_z1=-2.44,
    logphi1_z2=-3.08,
    logphi1_z3=-4.14,
    logphi2_z1=-2.89,
    logphi2_z2=-3.29,
    logphi2_z3=-3.51,
    log10mstar_z1=10.79,
    log10mstar_z2=10.88,
    log10mstar_z3=10.84,
    alpha1=-0.28,
    alpha2=-1.48,
)


DEFAULT_DBL_SCHECHTER = OrderedDict(
    logphi1=-2.44,
    logphi2=-2.89,
    log10mstar=10.79,
    alpha1=-0.28,
    alpha2=-1.48,
)

DBL_SCHECHTER_BOUNDS = OrderedDict(
    logphi1=(-5.0, -1.5),
    logphi2=(-4.0, -2.0),
    log10mstar=(10.5, 11.25),
    alpha1=(-0.75, 0.0),
    alpha2=(-2.0, -1.0),
)


DEFAULT_DBL_SCHECHTER_FITTER = OrderedDict(
    logphi1=-2.44,
    delta_logphi1=-0.45,
    log10mstar=10.79,
    alpha1=-0.28,
    delta_alpha1=-1.2,
)

DBL_SCHECHTER_FITTER_BOUNDS = OrderedDict(
    logphi1=(-3.0, -2.0),
    delta_logphi1=(-0.65, -0.25),
    log10mstar=(10.7, 10.9),
    alpha1=(-0.4, -0.2),
    delta_alpha1=(-1.3, -1.1),
)


def leja19_continuity_model_smf(
    log10m,
    z,
    logphi1_z1=None,
    logphi1_z2=None,
    logphi1_z3=None,
    logphi2_z1=None,
    logphi2_z2=None,
    logphi2_z3=None,
    log10mstar_z1=None,
    log10mstar_z2=None,
    log10mstar_z3=None,
    alpha1=None,
    alpha2=None,
):
    """Stellar mass function taken from Leja+19,
    https://arxiv.org/abs/1910.04168.

    Parameters
    ----------
    log10m : float or ndarray of shape (npts, )
        Base-10 log of stellar mass assuming h=0.697

    z : float or ndarray of shape (npts, )

    logphiN_zM : float, optional
        normalization of the N=1 or N=2 Schechter function
        evaluated at zM = {z1=0.2, z2=1.6, z3=3.0}.

    log10mstar_zM : float, optional
        Characteristic stellar mass of the Schechter function
        evaluated at zM = {z1=0.2, z2=1.6, z3=3.0}.

    alphaN : float, optional
        Power-law index of the N=1,2 Schechter functions

    Returns
    -------
    phi : ndarray of shape (npts, )
        Units are 1/Mpc**3/dex
    """

    log10m, z = _get_1d_arrays(log10m, z)
    leja19_params = _get_leja19_params(
        logphi1_z1,
        logphi1_z2,
        logphi1_z3,
        logphi2_z1,
        logphi2_z2,
        logphi2_z3,
        log10mstar_z1,
        log10mstar_z2,
        log10mstar_z3,
        alpha1,
        alpha2,
    )
    smf_params_at_z = _get_dbl_shechter_params_at_z(leja19_params, z)
    logphi1, logphi2, log10mstar, alpha1, alpha2 = smf_params_at_z
    return _double_schechter(log10m, logphi1, logphi2, log10mstar, alpha1, alpha2)


def double_schechter(logsm, zobs=ZOBS, **kwargs):
    pdict = _get_dbl_schechter_param_dict(zobs=zobs, **kwargs)
    dbl_shechter_params = np.array(list(pdict.values()))
    return np.array(_double_schechter(logsm, *dbl_shechter_params))


def _get_dbl_schechter_param_dict(zobs=ZOBS, **kwargs):
    d = DEFAULT_PARAMS
    pdict = OrderedDict([(key, kwargs.get(key, d.get(key))) for key in d.keys()])
    leja19_params = np.array(list(pdict.values()))
    dbl_shechter_params = _get_dbl_shechter_params_at_z(leja19_params, zobs)
    logphi1, logphi2, log10mstar, alpha1, alpha2 = dbl_shechter_params
    dbl_schechter_param_dict = OrderedDict(
        logphi1=kwargs.get("logphi1", logphi1),
        logphi2=kwargs.get("logphi2", logphi2),
        log10mstar=kwargs.get("log10mstar", log10mstar),
        alpha1=kwargs.get("alpha1", alpha1),
        alpha2=kwargs.get("alpha2", alpha2),
    )
    return dbl_schechter_param_dict


@jjit
def _double_schechter(log10m, logphi1, logphi2, log10mstar, alpha1, alpha2):
    s1 = _schechter(log10m, logphi1, log10mstar, alpha1)
    s2 = _schechter(log10m, logphi2, log10mstar, alpha2)
    return s1 + s2


@jjit
def _double_schechter_fitter(
    log10m, logphi1, delta_logphi1, log10mstar, alpha1, delta_alpha1
):
    logphi2 = logphi1 + delta_logphi1
    alpha2 = alpha1 + delta_alpha1
    return _double_schechter(log10m, logphi1, logphi2, log10mstar, alpha1, alpha2)


@jjit
def _schechter(log10m, logphi, log10mstar, alpha):
    A = jnp.log(10) * 10 ** logphi
    x = log10m - log10mstar
    phi = A * 10 ** (x * (alpha + 1)) * jnp.exp(-(10 ** x))
    return phi


def _get_dbl_shechter_params_at_z(leja19_params, z):
    logphi1 = _parameter_at_z(*leja19_params[0:3], z)
    logphi2 = _parameter_at_z(*leja19_params[3:6], z)
    log10mstar = _parameter_at_z(*leja19_params[6:9], z)
    alpha1, alpha2 = leja19_params[-2:]
    return logphi1, logphi2, log10mstar, alpha1, alpha2


def _get_leja19_params(
    logphi1_z1,
    logphi1_z2,
    logphi1_z3,
    logphi2_z1,
    logphi2_z2,
    logphi2_z3,
    log10mstar_z1,
    log10mstar_z2,
    log10mstar_z3,
    alpha1,
    alpha2,
):
    if logphi1_z1 is None:
        logphi1_z1 = DEFAULT_PARAMS["logphi1_z1"]
    if logphi1_z2 is None:
        logphi1_z2 = DEFAULT_PARAMS["logphi1_z2"]
    if logphi1_z3 is None:
        logphi1_z3 = DEFAULT_PARAMS["logphi1_z3"]
    if logphi2_z1 is None:
        logphi2_z1 = DEFAULT_PARAMS["logphi2_z1"]
    if logphi2_z2 is None:
        logphi2_z2 = DEFAULT_PARAMS["logphi2_z2"]
    if logphi2_z3 is None:
        logphi2_z3 = DEFAULT_PARAMS["logphi2_z3"]
    if log10mstar_z1 is None:
        log10mstar_z1 = DEFAULT_PARAMS["log10mstar_z1"]
    if log10mstar_z2 is None:
        log10mstar_z2 = DEFAULT_PARAMS["log10mstar_z2"]
    if log10mstar_z3 is None:
        log10mstar_z3 = DEFAULT_PARAMS["log10mstar_z3"]
    if alpha1 is None:
        alpha1 = DEFAULT_PARAMS["alpha1"]
    if alpha2 is None:
        alpha2 = DEFAULT_PARAMS["alpha2"]
    leja19_params = (
        logphi1_z1,
        logphi1_z2,
        logphi1_z3,
        logphi2_z1,
        logphi2_z2,
        logphi2_z3,
        log10mstar_z1,
        log10mstar_z2,
        log10mstar_z3,
        alpha1,
        alpha2,
    )
    return leja19_params


def _parameter_at_z(y1, y2, y3, z, z1=0.2, z2=1.6, z3=3.0):
    a = ((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (
        z3 ** 2 - z1 ** 2 + (z2 ** 2 - z1 ** 2) / (z2 - z1) * (z1 - z3)
    )
    b = ((y2 - y1) - a * (z2 ** 2 - z1 ** 2)) / (z2 - z1)
    c = y1 - a * z1 ** 2 - b * z1
    return a * z ** 2 + b * z + c


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
