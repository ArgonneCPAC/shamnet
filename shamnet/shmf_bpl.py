"""
"""
from collections import OrderedDict
from copy import deepcopy
from jax import jit as jax_jit
from jax import numpy as jax_np

DEFAULT_PARAMS = OrderedDict(
    c0_y0=-2.54,
    c1_y0=-0.074,
    c2_y0=-0.1,
    c0_m=-1.02,
    c1_m=-0.13,
    c2_m=-0.03,
    c0_xc=15.1,
    c1_xc=-0.54,
    x0=12.25,
    kc=3,
    dy=5,
)


def log10_cumulative_shmf_bpl(logmp, z, **kwargs):
    """Cumulative subhalo mass function vs. Mpeak and z.

    Parameters
    ----------
    logmpeak : ndarray of shape (n, )
        Base-10 log of halo mass in Msun, assuming h=1.

    z : float

    **params : float, optional
        All keywords of DEFAULT_PARAMS dictionary are accepted

    Returns
    -------
    lognd : ndarray of shape (n, )
        Base-10 log of cumulative number density of halos more massive than logmpeak,
        in units of 1/Mpc**3, comoving with h=1.

    """
    d = deepcopy(DEFAULT_PARAMS)
    d.update(kwargs)
    c0_y0, c1_y0, c2_y0, c0_m, c1_m, c2_m, c0_xc, c1_xc, x0, kc, dy = list(d.values())
    y0 = _get_y0(z, c0_y0, c1_y0, c2_y0)
    m = _get_m(z, c0_m, c1_m, c2_m)
    xc = _get_xc(z, c0_xc, c1_xc)
    return _log10_cumulative_shmf_prediction(logmp, y0, m, xc, x0, kc, dy)


def _get_y0(z, c0_y0=-2.54, c1_y0=-0.074, c2_y0=-0.1):
    return c0_y0 + c1_y0 * z + c2_y0 * z ** 2


def _get_m(z, c0_m=-1.02, c1_m=-0.13, c2_m=-0.03):
    return c0_m + c1_m * z + c2_m * z ** 2


def _get_xc(z, c0_xc=15.1, c1_xc=-0.54):
    return c0_xc + c1_xc * z


@jax_jit
def _jax_sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jax_np.exp(-k * (x - x0)))


@jax_jit
def _log10_cumulative_shmf_prediction(logmp, y0, m, xc, x0, kc, dy):
    y = y0 + m * (logmp - x0)
    return _jax_sigmoid(logmp, xc, kc, y, y - dy)
