"""Functions used to generate training data for the SHAMnet initializer.
The get_all_smhm_params_from_varied_params function defines which SMHM params are free.
The fit_smhm_params function identifies the optimal SMHM for the input SMF.
The generate_traindata_batch function formats the results into the appropriate matrices.
"""
from jax import numpy as jnp
from jax import jit as jjit, vmap
from jax import random as jran
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from .utils import jax_adam_wrapper
from .jaxnn_helpers import get_randomly_spaced_array
from .shamnet_traindata import log_smhm_likelihood
from .shamnet_traindata import DEFAULT_SMHM_PARAMS, DEFAULT_SMHM_U_PARAMS
from .shamnet_traindata import _bounded_params_from_unbounded
from .shamnet_traindata import DEFAULT_SMHM_PARAM_BOUNDS
from .shamnet_traindata import _PBOUND_X0, _PBOUND_K, _sigmoid
from .sample_shamnet_param_space import SCATTER_MINS, SCATTER_MAXS
from .sample_shamnet_param_space import LHS_SIGMA, SAMPLER_MEAN
from .sample_shamnet_param_space import sample_shamnet_params, LOG10_SMF_CLIP


FIXED_SMHM_KEYS = ("smhm_k_logm", "smhm_lowm_index_k", "smhm_highm_index_k")
INDX_VARIED_PARAMS = [
    list(DEFAULT_SMHM_PARAMS).index(a)
    for a in DEFAULT_SMHM_PARAMS.keys()
    if a not in FIXED_SMHM_KEYS
]
LOGMH_TABLE = jnp.linspace(10, 16, 500)
SMF_LOGSM_BINS = jnp.linspace(9, 11.9, 100)
SMF_LO_BOUNDS = SAMPLER_MEAN - LHS_SIGMA
SMF_HI_BOUNDS = SAMPLER_MEAN + LHS_SIGMA
X_MINS = jnp.array([*SMF_LO_BOUNDS, *SCATTER_MINS, 6.0])
X_MAXS = jnp.array([*SMF_HI_BOUNDS, *SCATTER_MAXS, 16.5])
FIXED_REDSHIFT = 0.0
SHAMNET_LOSS_LOGMH_TABLE = jnp.linspace(11, 15.5, 200)


@jjit
def get_all_smhm_params_from_varied_params(
    varied_smhm_params,
    smhm_logm_crit=DEFAULT_SMHM_PARAMS["smhm_logm_crit"],
    smhm_ratio_logm_crit=DEFAULT_SMHM_PARAMS["smhm_ratio_logm_crit"],
    smhm_k_logm=DEFAULT_SMHM_PARAMS["smhm_k_logm"],
    smhm_lowm_index_x0=DEFAULT_SMHM_PARAMS["smhm_lowm_index_x0"],
    smhm_lowm_index_k=DEFAULT_SMHM_PARAMS["smhm_lowm_index_k"],
    smhm_lowm_index_ylo=DEFAULT_SMHM_PARAMS["smhm_lowm_index_ylo"],
    smhm_lowm_index_yhi=DEFAULT_SMHM_PARAMS["smhm_lowm_index_yhi"],
    smhm_highm_index_x0=DEFAULT_SMHM_PARAMS["smhm_highm_index_x0"],
    smhm_highm_index_k=DEFAULT_SMHM_PARAMS["smhm_highm_index_k"],
    smhm_highm_index_ylo=DEFAULT_SMHM_PARAMS["smhm_highm_index_ylo"],
    smhm_highm_index_yhi=DEFAULT_SMHM_PARAMS["smhm_highm_index_yhi"],
):
    (
        smhm_logm_crit,
        smhm_ratio_logm_crit,
        smhm_lowm_index_x0,
        smhm_lowm_index_ylo,
        smhm_lowm_index_yhi,
        smhm_highm_index_x0,
        smhm_highm_index_ylo,
        smhm_highm_index_yhi,
    ) = varied_smhm_params

    all_smhm_params = (
        smhm_logm_crit,
        smhm_ratio_logm_crit,
        smhm_k_logm,
        smhm_lowm_index_x0,
        smhm_lowm_index_k,
        smhm_lowm_index_ylo,
        smhm_lowm_index_yhi,
        smhm_highm_index_x0,
        smhm_highm_index_k,
        smhm_highm_index_ylo,
        smhm_highm_index_yhi,
    )

    return all_smhm_params


@jjit
def get_all_params_from_varied_uparams(
    varied_u_params,
    smhm_logm_crit=DEFAULT_SMHM_U_PARAMS["smhm_logm_crit"],
    smhm_ratio_logm_crit=DEFAULT_SMHM_U_PARAMS["smhm_ratio_logm_crit"],
    smhm_k_logm=DEFAULT_SMHM_U_PARAMS["smhm_k_logm"],
    smhm_lowm_index_x0=DEFAULT_SMHM_U_PARAMS["smhm_lowm_index_x0"],
    smhm_lowm_index_k=DEFAULT_SMHM_U_PARAMS["smhm_lowm_index_k"],
    smhm_lowm_index_ylo=DEFAULT_SMHM_U_PARAMS["smhm_lowm_index_ylo"],
    smhm_lowm_index_yhi=DEFAULT_SMHM_U_PARAMS["smhm_lowm_index_yhi"],
    smhm_highm_index_x0=DEFAULT_SMHM_U_PARAMS["smhm_highm_index_x0"],
    smhm_highm_index_k=DEFAULT_SMHM_U_PARAMS["smhm_highm_index_k"],
    smhm_highm_index_ylo=DEFAULT_SMHM_U_PARAMS["smhm_highm_index_ylo"],
    smhm_highm_index_yhi=DEFAULT_SMHM_U_PARAMS["smhm_highm_index_yhi"],
):
    (
        smhm_logm_crit,
        smhm_ratio_logm_crit,
        smhm_lowm_index_x0,
        smhm_lowm_index_ylo,
        smhm_lowm_index_yhi,
        smhm_highm_index_x0,
        smhm_highm_index_ylo,
        smhm_highm_index_yhi,
    ) = varied_u_params

    all_u_params = (
        smhm_logm_crit,
        smhm_ratio_logm_crit,
        smhm_k_logm,
        smhm_lowm_index_x0,
        smhm_lowm_index_k,
        smhm_lowm_index_ylo,
        smhm_lowm_index_yhi,
        smhm_highm_index_x0,
        smhm_highm_index_k,
        smhm_highm_index_ylo,
        smhm_highm_index_yhi,
    )
    return _bounded_params_from_unbounded(all_u_params)


get_all_params_from_varied_uparams_batch = jjit(
    vmap(get_all_params_from_varied_uparams)
)


@jjit
def mse_loss(varied_unbounded_params, data):
    all_smhm_u_params = get_all_params_from_varied_uparams(varied_unbounded_params)

    return log_smhm_likelihood(all_smhm_u_params, *data)


def fit_smhm_params(smf_target_params, shmf_params, scatter_params, n_steps, n_warmup):
    """Fit the input SMF parameters with a SMHM"""
    loss_data = (
        smf_target_params,
        shmf_params,
        scatter_params,
        LOGMH_TABLE,
        SMF_LOGSM_BINS,
        LOG10_SMF_CLIP,
    )

    fixed_smf_p = OrderedDict()
    fixed_shmf_p = OrderedDict()
    fixed_scatter_p = OrderedDict()
    fixed_smhm_p = OrderedDict()
    smhm_dict = deepcopy(DEFAULT_SMHM_U_PARAMS)
    for key in FIXED_SMHM_KEYS:
        uval = smhm_dict.pop(key)
        _bounds = DEFAULT_SMHM_PARAM_BOUNDS[key]
        fixed_smhm_p[key] = _sigmoid(uval, _PBOUND_X0, _PBOUND_K, *_bounds)
    up0 = np.array(list(smhm_dict.values()))

    _ures = jax_adam_wrapper(
        mse_loss, up0, loss_data, n_steps, n_warmup=n_warmup, step_size=0.01
    )
    if n_warmup > 0:
        final_n_warmup = 1
    else:
        final_n_warmup = 0
    ures = jax_adam_wrapper(
        mse_loss, _ures[0], loss_data, n_steps, n_warmup=final_n_warmup, step_size=0.002
    )
    _smhm_u_best, loss = ures[:2]
    all_smhm_best = get_all_params_from_varied_uparams(_smhm_u_best)
    fixed_params = fixed_smf_p, fixed_shmf_p, fixed_scatter_p, fixed_smhm_p

    smhm_best = [all_smhm_best[i] for i in INDX_VARIED_PARAMS]
    return smhm_best, all_smhm_best, loss, fixed_params


def generate_traindata_batch(n_batch, n_steps, n_warmup, ran_key):
    """Primary function used to generate collections of best-fit SMHMs that will be
    written to hdf5 for later use as training data for the SHAMnet initializer."""
    all_params = sample_shamnet_params(ran_key, n_batch)
    smf_param_batch, shmf_param_batch, scatter_param_batch = all_params

    X_collector = []
    target_collector = []
    loss_collector = []
    gen = zip(smf_param_batch, shmf_param_batch, scatter_param_batch)
    for batch in gen:
        smf_target_params, shmf_params, scatter_params = batch
        args = smf_target_params, shmf_params, scatter_params, n_steps, n_warmup
        _ret = fit_smhm_params(*args)
        varied_smhm_best, all_smhm_best, loss, fixed_params = _ret
        target_collector.append(varied_smhm_best)
        loss_collector.append(loss)
        X = (*smf_target_params, *scatter_params)
        X_collector.append(X)

    traindata_X = np.array(X_collector)
    traindata_Y = np.array(target_collector)
    traindata_loss = np.array(loss_collector)

    all_smhm_mins = np.array(
        [DEFAULT_SMHM_PARAM_BOUNDS[key][0] for key in DEFAULT_SMHM_PARAM_BOUNDS.keys()]
    )
    all_smhm_maxs = np.array(
        [DEFAULT_SMHM_PARAM_BOUNDS[key][1] for key in DEFAULT_SMHM_PARAM_BOUNDS.keys()]
    )
    smhm_mins_varied = [all_smhm_mins[i] for i in INDX_VARIED_PARAMS]
    smhm_maxs_varied = [all_smhm_maxs[i] for i in INDX_VARIED_PARAMS]
    X_bounds = X_MINS, X_MAXS
    Y_bounds = smhm_mins_varied, smhm_maxs_varied
    return traindata_X, traindata_Y, traindata_loss, fixed_params, X_bounds, Y_bounds


def shamnet_loss_data_generator(
    ran_key, batch_size, n_smf_bins, lg_smf_lo, lg_smf_hi, lupton_clip
):
    """
    SHAMnet training data generator for SMF-based loss

    Parameters
    ----------
    ran_key : jax.random.PRNGKey(seed)

    batch_size : int

    n_smf_bins: int
        Number of bins used for the target SMF

    lg_smf_lo : float
        Lower bound in base-10 log of Mstar for the target SMF

    lg_smf_hi : float
        Upper bound in base-10 log of Mstar for the target SMF

    lupton_clip : float
        Cutoff parameter when taking the base-10 log of the SMF

    """
    n_mh = SHAMNET_LOSS_LOGMH_TABLE.size
    logmh_lo, logmh_hi = SHAMNET_LOSS_LOGMH_TABLE.min(), SHAMNET_LOSS_LOGMH_TABLE.max()
    while True:
        ran_key, mh_key, smf_bins_key, param_key = jran.split(ran_key, 4)
        all_params = sample_shamnet_params(param_key, batch_size)
        smf_batch, scatter_batch = all_params[0], all_params[2]
        logmh_table = get_randomly_spaced_array(mh_key, n_mh, logmh_lo, logmh_hi)
        logmh_batch = jnp.tile(logmh_table, batch_size).reshape((batch_size, n_mh))
        smf_logsm_bins = get_randomly_spaced_array(
            smf_bins_key, n_smf_bins, lg_smf_lo, lg_smf_hi
        )
        yield smf_batch, scatter_batch, logmh_batch, smf_logsm_bins, lupton_clip
