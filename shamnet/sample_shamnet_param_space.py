"""Function used to sample the parameter space of SHAMNet"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from jax import random as jran
import numpy as np
from .shmf_planck import _get_lognd_at_logm_crit_at_z, _get_logm_crit_at_z
from .shamnet_traindata import _schechter
from .latin_hypercube import latin_hypercube_from_cov

DEFAULT_SCHECHTER_PARAMS = OrderedDict(logphi=-2.3, alpha=-1.06)
LOGSM_TARGET = np.linspace(9, 12, 100)
LOG_SMF_TOL = 0.3
SAMPLER_COV = np.array([[0.014, 0.0075], [0.0075, 0.015]])
SAMPLER_MEAN = np.array([-2.3, -1.06])

SCATTER_MINS = np.array((0.2, 0.2))
SCATTER_MAXS = np.array((0.5, 0.5))
Z_TARGET_SMF = 0.05
LHS_SIGMA = 5.0
LOG10_SMF_CLIP = -14.0


def log_likelihood(theta, target_smf, target_lgsm, yerr):
    pred_smf = _schechter(target_lgsm, *theta)
    yerr_sq = yerr ** 2
    diff_smf = target_smf - pred_smf
    return -0.5 * jnp.sum(diff_smf ** 2 / yerr_sq + jnp.log(yerr_sq))


_schechter_vmap = jjit(vmap(_schechter, in_axes=(None, 0, 0)))


def _mask_bad_params(samples, tol):
    # Enforce SMFs are monotonic
    logphi, alpha = samples[:, 0], samples[:, 1]
    log_smf_samples = np.log10(_schechter_vmap(LOGSM_TARGET, logphi, alpha))
    delta_log_smf_samples = np.diff(log_smf_samples, axis=1)
    monotonic_msk = np.all(delta_log_smf_samples < 0, axis=1)

    # Enforce agreement with fiducial SMF
    smf_target = _schechter(LOGSM_TARGET, *DEFAULT_SCHECHTER_PARAMS.values())
    log_smf_target = np.log10(smf_target)
    log_smf_diff = log_smf_samples - log_smf_target.reshape((1, -1))
    close_to_target_msk = np.all(np.abs(log_smf_diff) < tol, axis=1)

    return monotonic_msk & close_to_target_msk


def _generate_lhs_samples(n, sig):
    return latin_hypercube_from_cov(SAMPLER_MEAN, SAMPLER_COV, sig, n)


def _good_schechter_params(n, tol, sig):
    proposed_samples = _generate_lhs_samples(n, sig)
    msk = _mask_bad_params(proposed_samples, tol)
    good_params = proposed_samples[msk]
    return good_params


def generate_schechter_param_samples(n, tol=LOG_SMF_TOL, sig=LHS_SIGMA):
    param_samples = _good_schechter_params(n, tol, sig)
    n_good_params = param_samples.shape[0]

    while n_good_params < n:
        new_params = _good_schechter_params(n, tol, sig)
        param_samples = np.vstack((param_samples, new_params))
        n_good_params = param_samples.shape[0]
    return param_samples[:n, :]


def sample_shamnet_params(ran_key, n_sample, tol=LOG_SMF_TOL, sig=LHS_SIGMA):
    shmf_param_collector = []
    scatter_collector = []
    smf_params = generate_schechter_param_samples(n_sample, tol=tol, sig=sig)
    for i in range(n_sample):
        keys = jran.split(ran_key, 4)
        ran_key_smf, scat_lo_key, scat_hi_key, ran_key = keys
        scat_lo = jran.uniform(
            scat_lo_key, minval=SCATTER_MINS[0], maxval=SCATTER_MAXS[0]
        )
        scat_hi = jran.uniform(
            scat_hi_key, minval=SCATTER_MINS[1], maxval=SCATTER_MAXS[1]
        )
        scatter_params = scat_lo, scat_hi
        scatter_collector.append(scatter_params)
        logm_crit = _get_logm_crit_at_z(Z_TARGET_SMF)
        lognd_at_logm_crit = _get_lognd_at_logm_crit_at_z(Z_TARGET_SMF)
        shmf_params = jnp.array((lognd_at_logm_crit, logm_crit))

        shmf_param_collector.append(shmf_params)
    shmf_params = jnp.array(shmf_param_collector)
    scatter_params = jnp.array(scatter_collector)
    return smf_params, shmf_params, scatter_params
