"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp


K_FIXED = 0.9
LOWM_FIXED = -1.4
HIGHM_FIXED = -5.25

DEFAULT_SHMF_PARAMS = OrderedDict(lognd_at_logm_crit=-7.37, logm_crit=15.16)
DEFAULT_SHMF_PARAM_BOUNDS = OrderedDict(
    lognd_at_logm_crit=(-8.0, -5.0), logm_crit=(13.0, 15.5)
)
Z0P5_PARAMS = OrderedDict(lognd_at_logm_crit=-7.0, logm_crit=14.75)
Z1P0_PARAMS = OrderedDict(lognd_at_logm_crit=-6.61, logm_crit=14.31)
Z2P0_PARAMS = OrderedDict(lognd_at_logm_crit=-6.16, logm_crit=13.66)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _shmf_plaw_index(logmh, lgm0):
    return _sigmoid(logmh, lgm0, K_FIXED, LOWM_FIXED, HIGHM_FIXED)


@jjit
def _log10_cumulative_shmf(logmh, lognd_at_logm_crit, logm_crit):
    powerlaw_index = _shmf_plaw_index(logmh, logm_crit)
    return lognd_at_logm_crit + powerlaw_index * (logmh - logm_crit)


@jjit
def _calculate_lgnd_from_lgcnd(lgcnd, lgxbins):
    """Calculate differential number density from cumulative."""
    return jnp.log10(-jnp.diff(10 ** lgcnd)) - jnp.log10(jnp.diff(lgxbins))


@jjit
def _get_lognd_at_logm_crit_at_z(z):
    return _sigmoid(z, 0.5, 1.35, -8.1, -5.9)


@jjit
def _get_logm_crit_at_z(z):
    return _sigmoid(z, 0.75, 1, 16.175, 12.925)


@jjit
def _get_shmf_params_at_z(z):
    lognd_at_logm_crit = _get_lognd_at_logm_crit_at_z(z)
    logm_crit = _get_logm_crit_at_z(z)
    return jnp.array((lognd_at_logm_crit, logm_crit))
