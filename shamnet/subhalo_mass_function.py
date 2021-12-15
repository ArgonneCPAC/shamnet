"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp

DEFAULT_SHMF_PARAMS = OrderedDict(y0=-2.54, m=-1, xc=15.5, x0=12.25, kc=2.5, dy=9.5)
DEFAULT_SHMF_PARAM_BOUNDS = OrderedDict(
    y0=(-3, -2), m=(-2, -0.5), xc=(13, 17), x0=(11.5, 13.5), dy=(1, 20)
)


def log10_cumulative_shmf(logmp, y0=None, m=None, xc=None, x0=None, kc=None, dy=None):
    """Truncated power-law model for the cumulative subhalo mass function.

    Parameters
    ----------
    logmp : ndarray, shape (n, )
        Array stores base-10 log of halo mass

    params, optional
        In order: y0, m, xc, x0, kc, dy
        Defaults set in DEFAULT_SHMF_PARAMS

    Returns
    -------
    log10_cuml_nd : ndarray, shape (n, )
        Base-10 log of the cumulative subhalo abundance in units of Mpc^-3

    """
    y0, m, xc, x0, kc, dy = _get_all_shmf_params(y0, m, xc, x0, kc, dy)
    return _log10_cumulative_shmf(logmp, y0, m, xc, x0, kc, dy)


def _get_all_shmf_params(y0=None, m=None, xc=None, x0=None, kc=None, dy=None):
    """"""
    if y0 is None:
        y0 = DEFAULT_SHMF_PARAMS["y0"]
    if m is None:
        m = DEFAULT_SHMF_PARAMS["m"]
    if xc is None:
        xc = DEFAULT_SHMF_PARAMS["xc"]
    if x0 is None:
        x0 = DEFAULT_SHMF_PARAMS["x0"]
    if kc is None:
        kc = DEFAULT_SHMF_PARAMS["kc"]
    if dy is None:
        dy = DEFAULT_SHMF_PARAMS["dy"]
    return y0, m, xc, x0, kc, dy


@jjit
def _log10_cumulative_shmf(logmp, y0, m, xc, x0, kc, dy):
    """Differentiable kernel of the cumulative subhalo mass function."""
    y = y0 + m * (logmp - x0)
    return _jax_sigmoid(logmp, xc, kc, y, y - dy)


@jjit
def _calculate_lgnd_from_lgcnd(lgcnd, lgxbins):
    """Calculate differential number density from cumulative."""
    return jnp.log10(-jnp.diff(10 ** lgcnd)) - jnp.log10(jnp.diff(lgxbins))


@jjit
def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))
