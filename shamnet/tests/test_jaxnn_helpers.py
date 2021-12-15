"""
"""
from jax import random as jran
import numpy as np
from ..jaxnn_helpers import _unit_scale_traindata, _unscale_traindata
from ..leja19_smf import DBL_SCHECHTER_FITTER_BOUNDS


def test_rescaling_inverts():
    npts, ndim = 100_000, 2
    xmins = np.array((0.5, 7.5))
    dx = np.array((4.5, 8.5))
    xmaxs = xmins + dx

    ran_key = jran.PRNGKey(0)
    orig_traindata = np.zeros(shape=(npts, ndim))
    orig_traindata[:, 0] = dx[0] * jran.uniform(ran_key, (npts,)) + xmins[0]
    old_key, ran_key = jran.split(ran_key)
    orig_traindata[:, 1] = dx[1] * jran.uniform(ran_key, (npts,)) + xmins[1]
    assert np.all(orig_traindata[:, 0] >= xmins[0])
    assert np.all(orig_traindata[:, 1] >= xmins[1])
    assert np.all(orig_traindata[:, 0] <= xmaxs[0])
    assert np.all(orig_traindata[:, 1] <= xmaxs[1])

    scaled_traindata = _unit_scale_traindata(orig_traindata, xmins, xmaxs)
    assert np.all(scaled_traindata[:, 0] >= 0)
    assert np.all(scaled_traindata[:, 1] >= 0)
    assert np.all(scaled_traindata[:, 0] <= 1)
    assert np.all(scaled_traindata[:, 1] <= 1)

    derived_traindata = _unscale_traindata(scaled_traindata, xmins, xmaxs)
    assert np.allclose(derived_traindata, orig_traindata, rtol=1e-5)


def test_smf_training_data_generation_scaling():
    xmins = [x[0] for x in DBL_SCHECHTER_FITTER_BOUNDS.values()]
    xmaxs = [x[1] for x in DBL_SCHECHTER_FITTER_BOUNDS.values()]
    ran_key = jran.PRNGKey(0)
    old_key, ran_key = jran.split(ran_key)

    npts = 500
    u = jran.uniform(ran_key, (npts, len(xmins)))
    smf_param_batch = _unscale_traindata(u, xmins, xmaxs)
    for ibatch in range(npts):
        smf_fitter_params = smf_param_batch[ibatch, :]
        for ip, bounds in enumerate(DBL_SCHECHTER_FITTER_BOUNDS.values()):
            lo, hi = bounds
            assert lo < smf_fitter_params[ip] < hi
