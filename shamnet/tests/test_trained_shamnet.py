"""
"""
import os
from jax import random as jran
from ..sham_network import get_sham_network
from ..generate_shamnet_traindata import shamnet_loss_data_generator
from ..shamnet_traindata import predict_smf_from_smf_param_batch
from ..load_shamnet_traindata import load_initializer_tdata_and_vdata
from ..utils import lupton_log10
import numpy as np


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_shamnet_smf_predictions_agree_with_previously_unseen_smfs():
    ran_key = jran.PRNGKey(0)
    res = get_sham_network()
    net_init, _predict_logsm_from_X = res[:2]
    smhm_batch_loss, smf_batch_loss = res[2:4]
    pred_logsm_batch, predict_smf_batch = res[6:8]

    lg_smf_lo, lg_smf_hi, n_smf_bins = 8.9, 12.05, 100
    smf_logsm_bins = np.linspace(lg_smf_lo, lg_smf_hi, n_smf_bins)
    clip = -15

    shamnet_params = net_init()
    batch_size = 100

    num_vbatches = 10
    args = ran_key, batch_size, n_smf_bins, lg_smf_lo, lg_smf_hi, clip
    gen = shamnet_loss_data_generator(*args)
    for iv in range(num_vbatches):
        loss_data = next(gen)
        smf_vbatch, scatter_vbatch, logmh_vbatch = loss_data[:3]

        res = predict_smf_batch(
            shamnet_params, smf_logsm_bins, smf_vbatch, scatter_vbatch, logmh_vbatch
        )
        logsm_smf_vpred, smf_vpred = res
        smf_vtargets = predict_smf_from_smf_param_batch(logsm_smf_vpred, smf_vbatch)
        logdiff = lupton_log10(smf_vpred, clip) - lupton_log10(smf_vtargets, clip)

        mu, std = np.mean(logdiff, axis=0), np.std(logdiff, axis=0)
        assert np.all(np.abs(mu) < 0.1)
        assert np.all(std < 0.05)


def test_shamnet_smf_predictions_agree_with_traindata_smfs():
    bn = "SHAMnet11_tdata_SHAMnet11_tdata_SHAMnet11_testing_tdata.h5"
    tdata_fn = os.path.join(_THIS_DRNAME, "testing_data", bn)
    ran_key = jran.PRNGKey(0)
    n_train, n_validate = 50, 15
    train_key, ran_key = jran.split(ran_key)
    tdata, vdata = load_initializer_tdata_and_vdata(
        tdata_fn,
        n_train,
        train_key,
        10,
        15.5,
        n_mh=300,
        tol=-1.5,
        n_validate=n_validate,
    )
    X_tdata, Y_tdata, smf_tdata, scatter_tdata = tdata[:4]
    logmh_tdata, smhm_tdata, fixed_shmf_params = tdata[4:]
    X_vdata, Y_vdata, smf_vdata, scatter_vdata = vdata[:4]
    logmh_vdata, smhm_vdata, __ = vdata[4:]

    res = get_sham_network()
    net_init, _predict_logsm_from_X = res[:2]
    smhm_batch_loss, smf_batch_loss = res[2:4]
    pred_logsm_batch, predict_smf_batch = res[6:8]
    shamnet_params = net_init()

    lg_smf_lo, lg_smf_hi, n_smf_bins = 8.9, 12.05, 100
    smf_logsm_bins = np.linspace(lg_smf_lo, lg_smf_hi, n_smf_bins)
    clip = -15

    # Test performance on tdata
    logsm_smf_tpred, smf_tpred = predict_smf_batch(
        shamnet_params, smf_logsm_bins, smf_tdata, scatter_tdata, logmh_tdata
    )
    smf_ttargets = predict_smf_from_smf_param_batch(logsm_smf_tpred, smf_tdata)
    logdiff = lupton_log10(smf_tpred, clip) - lupton_log10(smf_ttargets, clip)
    mu, std = np.mean(logdiff, axis=0), np.std(logdiff, axis=0)
    assert np.all(np.abs(mu) < 0.1)
    assert np.all(std < 0.05)

    # Test performance on vdata
    logsm_smf_vpred, smf_vpred = predict_smf_batch(
        shamnet_params, smf_logsm_bins, smf_vdata, scatter_vdata, logmh_vdata
    )
    smf_vtargets = predict_smf_from_smf_param_batch(logsm_smf_vpred, smf_vdata)
    logdiff = lupton_log10(smf_vpred, clip) - lupton_log10(smf_vtargets, clip)
    mu, std = np.mean(logdiff, axis=0), np.std(logdiff, axis=0)
    assert np.all(np.abs(mu) < 0.1)
    assert np.all(std < 0.05)
