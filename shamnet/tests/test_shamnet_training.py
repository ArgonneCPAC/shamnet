"""
"""
import os
from ..sample_shamnet_param_space import sample_shamnet_params
from jax import random as jran
from ..sample_shamnet_param_space import DEFAULT_SCHECHTER_PARAMS, _schechter
from ..sample_shamnet_param_space import LOG_SMF_TOL, LOGSM_TARGET, _schechter_vmap
import numpy as np
from jax.experimental import optimizers as jax_opt
from ..sham_network import get_sham_network, _mse
from ..load_shamnet_traindata import load_initializer_tdata_and_vdata
from ..jaxnn_helpers import data_stream
from ..jaxnn_helpers import train_with_single_steps_per_batch
from ..generate_shamnet_traindata import shamnet_loss_data_generator
from ..shamnet_traindata import predict_smf_from_smf_param_batch


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_sample_shamnet_params_returns_reasonable_results():
    ran_key = jran.PRNGKey(0)
    n_samples = 1000
    ret = sample_shamnet_params(ran_key, n_samples)
    smf_params, shmf_params, scatter_params = ret

    logphi, alpha = smf_params[:, 0], smf_params[:, 1]
    sample_log_smfs = np.log10(_schechter_vmap(LOGSM_TARGET, logphi, alpha))

    p_fid = list(DEFAULT_SCHECHTER_PARAMS.values())
    target_log_smf = np.log10(_schechter(LOGSM_TARGET, *p_fid))
    for i in range(n_samples):
        sample_log_smf = sample_log_smfs[i, :]
        assert np.all(np.abs(sample_log_smf - target_log_smf) < LOG_SMF_TOL)
        delta_log_smf = np.diff(sample_log_smf)
        assert np.all(delta_log_smf < 0)


def test_shamnet_functions_evaluate():
    bn = "SHAMnet11_tdata_SHAMnet11_tdata_SHAMnet11_testing_tdata.h5"
    tdata_fn = os.path.join(_THIS_DRNAME, "testing_data", bn)
    _res = get_sham_network()
    net_init, net_pred, smhm_batch_loss = _res[:3]
    smf_batch_loss, smf_grads_batch_loss, smf_grads_joint_loss = _res[3:6]
    pred_logsm_batch, predict_smf_batch, predict_lnsmf_grads_batch = _res[6:9]

    ran_key = jran.PRNGKey(0)

    ran_key, p_key = jran.split(ran_key)
    init_net_params = net_init()

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

    batch_size = 10
    smhm_loss_data_generator = data_stream(batch_size, X_tdata, Y_tdata)
    loss_data = next(smhm_loss_data_generator)
    loss_init = smhm_batch_loss(init_net_params, loss_data)
    assert np.isfinite(loss_init)

    n_steps_tot = 5
    step_size = 0.01
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(init_net_params)
    args = (
        n_steps_tot,
        smhm_loss_data_generator,
        smhm_batch_loss,
        get_params,
        opt_state,
        opt_update,
    )
    _res = train_with_single_steps_per_batch(*args)
    opt_state, gd_step, loss_history, p_best, loss_min = _res

    LOG10_SMF_CLIP = -15
    N_SMF_BINS, SMF_BINS_LO, SMF_BINS_HI = 50, 8.5, 12.25

    loss_data_generator = shamnet_loss_data_generator(
        ran_key, batch_size, N_SMF_BINS, SMF_BINS_LO, SMF_BINS_HI, LOG10_SMF_CLIP
    )
    loss_data = next(loss_data_generator)
    smf_loss_init = smf_batch_loss(init_net_params, loss_data)
    assert np.isfinite(smf_loss_init)

    smf_grads_loss_init = smf_grads_batch_loss(init_net_params, loss_data)
    assert np.isfinite(smf_grads_loss_init), smf_grads_loss_init

    n_steps_tot = 3
    step_size = 0.001
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(p_best)

    args = (
        n_steps_tot,
        loss_data_generator,
        smf_batch_loss,
        get_params,
        opt_state,
        opt_update,
    )

    _res = train_with_single_steps_per_batch(*args, p_best=p_best)
    opt_state, gd_step, loss_history0, p_best, loss_min = _res

    smf_logsm_bins = np.linspace(9, 12, 50)
    logsm_smf_vpred, smf_vpreds = predict_smf_batch(
        p_best, smf_logsm_bins, smf_vdata, scatter_vdata, logmh_vdata
    )
    smf_vtargets = predict_smf_from_smf_param_batch(logsm_smf_vpred, smf_vdata)
    smf_vdata_loss = _mse(smf_vpreds, smf_vtargets)
    assert np.isfinite(smf_vdata_loss)

    ret = predict_lnsmf_grads_batch(
        p_best, smf_logsm_bins, smf_vdata, scatter_vdata, logmh_vdata, LOG10_SMF_CLIP
    )
    n_smf_params = 2
    correct_shape = (n_validate, smf_logsm_bins.size - 1, n_smf_params)
    assert correct_shape == ret.shape, "SMF grads have incorrect shape"

    loss_data = next(loss_data_generator)
