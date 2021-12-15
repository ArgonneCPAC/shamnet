"""Functions predicting Mstar and the SMF from the SMHM params."""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap, grad
from .shmf_planck import _calculate_lgnd_from_lgcnd, _log10_cumulative_shmf
from .smf_scatter_convolution import add_scatter_to_true_smf
from .utils import lupton_log10
from .leja19_smf import _schechter as _schechter_3params

LOG10MSTAR = 10.85

DEFAULT_SMHM_PARAMS = OrderedDict(
    smhm_logm_crit=11.35,
    smhm_ratio_logm_crit=-1.65,
    smhm_k_logm=1.6,
    smhm_lowm_index_x0=11.5,
    smhm_lowm_index_k=2,
    smhm_lowm_index_ylo=2.5,
    smhm_lowm_index_yhi=2.5,
    smhm_highm_index_x0=13.5,
    smhm_highm_index_k=2,
    smhm_highm_index_ylo=0.5,
    smhm_highm_index_yhi=0.5,
)
DEFAULT_SMHM_PARAM_BOUNDS = OrderedDict(
    smhm_logm_crit=(9.0, 16.0),
    smhm_ratio_logm_crit=(-5.0, 0.0),
    smhm_k_logm=(0.0, 25.0),
    smhm_lowm_index_x0=(9.0, 16.0),
    smhm_lowm_index_k=(0.0, 25.0),
    smhm_lowm_index_ylo=(0.0, 10.0),
    smhm_lowm_index_yhi=(0.0, 10.0),
    smhm_highm_index_x0=(9.0, 16.5),
    smhm_highm_index_k=(0.0, 15.0),
    smhm_highm_index_ylo=(-0.5, 15.0),
    smhm_highm_index_yhi=(-0.5, 15.0),
)

_PBOUND_X0, _PBOUND_K = 0.0, 0.1
BOUNDS = jnp.array(list(DEFAULT_SMHM_PARAM_BOUNDS.values()))
SCATTER_INFLECTION, SCATTER_K = 12.0, 1.0


@jjit
def _schechter(log10m, logphi, alpha):
    return _schechter_3params(log10m, logphi, LOG10MSTAR, alpha)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _scatter_model(logmh, scatter_dwarfs, scatter_clusters):
    return _sigmoid(
        logmh, SCATTER_INFLECTION, SCATTER_K, scatter_dwarfs, scatter_clusters
    )


@jjit
def _bounded_params_from_unbounded(up):
    p = [_sigmoid(up[i], _PBOUND_X0, _PBOUND_K, *BOUNDS[i]) for i in range(len(up))]
    return jnp.array(p)


@jjit
def _unbounded_params_from_bounded(p):
    up = [
        _inverse_sigmoid(p[i], _PBOUND_X0, _PBOUND_K, *BOUNDS[i]) for i in range(len(p))
    ]
    return jnp.array(up)


U_PARAMS = _unbounded_params_from_bounded(list(DEFAULT_SMHM_PARAMS.values()))
DEFAULT_SMHM_U_PARAMS = OrderedDict(
    [(s, v) for s, v in zip(DEFAULT_SMHM_PARAMS.keys(), U_PARAMS)]
)


@jjit
def _predict_no_scatter_smf(
    smhm_params,
    all_shmf_params,
    logmh_table,
):
    logsm_bins_smf = _logsm_from_logmh(smhm_params, logmh_table)

    # Compute cumulative abundance of input halos from the model
    log_cnd_halos = _log10_cumulative_shmf(logmh_table, *all_shmf_params)
    nd_halos = 10 ** _calculate_lgnd_from_lgcnd(log_cnd_halos, logmh_table)

    # Convert dn/dx to dn/dy
    logmh_table_mids = 0.5 * (logmh_table[:-1] + logmh_table[1:])
    dy_dx = _dlogsm_dlogmh_jacobian(smhm_params, logmh_table_mids)
    smf = nd_halos / dy_dx

    return logsm_bins_smf, smf


@jjit
def _logsm_from_logmh(smhm_params, logmh):
    """Kernel of the three-roll function mapping Mhalo ==> Mstar.

    Parameters
    ----------
    smhm_params : ndarray, shape (11, )
        Parameters of the three-roll function used to map Mhalo ==> Mstar,

    logmh : ndarray, shape (n, )
        Base-10 log of halo mass

    Returns
    -------
    logsm : ndarray, shape (n, )
        Base-10 log of stellar mass

    """
    logm_crit, log_sfeff_at_logm_crit, smhm_k_logm = smhm_params[0:3]
    lo_indx_pars = smhm_params[3:7]
    hi_indx_pars = smhm_params[7:11]

    lowm_index = _sigmoid(logmh, *lo_indx_pars)
    highm_index = _sigmoid(logmh, *hi_indx_pars)

    logsm_at_logm_crit = logm_crit + log_sfeff_at_logm_crit
    powerlaw_index = _sigmoid(logmh, logm_crit, smhm_k_logm, lowm_index, highm_index)

    return logsm_at_logm_crit + powerlaw_index * (logmh - logm_crit)


_dlogsm_dlogmh_jacobian = vmap(grad(_logsm_from_logmh, argnums=1), in_axes=(None, 0))
logsm_from_logmh_batch = jjit(vmap(_logsm_from_logmh, in_axes=(0, 0)))


@jjit
def predict_smf_from_smhm(
    shmf_params,
    scatter_params,
    smhm_params,
    logmh_table,
    logsm_bins_obs_smf,
):
    logsm_bins_true_smf, true_smf = _predict_no_scatter_smf(
        smhm_params, shmf_params, logmh_table
    )
    logmh_mids = 0.5 * (logmh_table[:-1] + logmh_table[1:])
    scatter = _scatter_model(logmh_mids, *scatter_params)
    args = logsm_bins_true_smf, true_smf, scatter, logsm_bins_obs_smf
    smf_prediction = add_scatter_to_true_smf(*args)
    logsm_binmids_obs_smf = 0.5 * (logsm_bins_obs_smf[:-1] + logsm_bins_obs_smf[1:])
    return logsm_binmids_obs_smf, smf_prediction


@jjit
def log_smhm_likelihood(
    smhm_params,
    smf_params,
    shmf_params,
    scatter_params,
    logmh_table,
    logsm_bins_obs_smf,
    log10_smf_clip,
):
    logsm_smf_pred, smf_pred = predict_smf_from_smhm(
        shmf_params, scatter_params, smhm_params, logmh_table, logsm_bins_obs_smf
    )
    smf_target = _schechter(logsm_smf_pred, *smf_params)
    log10_smf_pred = lupton_log10(smf_pred, log10_smf_clip)
    log10_smf_target = lupton_log10(smf_target, log10_smf_clip)
    loss = _mse(log10_smf_pred, log10_smf_target)
    return loss


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def _predict_smf_from_smhm_kernel(
    all_smhm_params, shmf_params, scatter_params, logmh_table, logsm_bins_obs_smf
):
    logsm_smf_pred, smf_pred = predict_smf_from_smhm(
        shmf_params, scatter_params, all_smhm_params, logmh_table, logsm_bins_obs_smf
    )
    return logsm_smf_pred, smf_pred


_a = (0, None, 0, None, None)
_predict_smf_from_smhm_batch = vmap(
    _predict_smf_from_smhm_kernel, in_axes=_a, out_axes=(None, 0)
)


@jjit
def predict_smf_from_smhm_batch(
    all_smhm_params,
    shmf_params,
    scatter_params,
    logmh_table,
    logsm_bins_obs_smf,
):
    return _predict_smf_from_smhm_batch(
        all_smhm_params, shmf_params, scatter_params, logmh_table, logsm_bins_obs_smf
    )


@jjit
def predict_smf_from_params(logsm, smf_params):
    return _schechter(logsm, *smf_params)


predict_smf_from_smf_param_batch = jjit(
    vmap(predict_smf_from_params, in_axes=(None, 0))
)


@jjit
def _predict_lnsmf_from_params(logsm, smf_params, log10_clip):
    return lupton_log10(_schechter(logsm, *smf_params), log10_clip)


_predict_lnsmf_grads_singlepoint = grad(_predict_lnsmf_from_params, argnums=1)

predict_ln_smf_grads_from_params = jjit(
    vmap(
        vmap(_predict_lnsmf_grads_singlepoint, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )
)
