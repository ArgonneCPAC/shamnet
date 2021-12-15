"""
"""
import os
from collections import namedtuple
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax import vmap, grad, value_and_grad
from .jaxnn_helpers import get_network, load_network_from_h5
from .jaxnn_helpers import _unscale_traindata, _unit_scale_traindata
from .shmf_planck import _calculate_lgnd_from_lgcnd, _log10_cumulative_shmf
from .shmf_planck import _get_shmf_params_at_z
from .generate_shamnet_traindata import X_MINS, X_MAXS, FIXED_REDSHIFT
from .shamnet_traindata import _scatter_model, predict_smf_from_smf_param_batch
from .shamnet_traindata import predict_ln_smf_grads_from_params
from .smf_scatter_convolution import add_scatter_to_true_smf
from .smf_scatter_convolution import _add_scatter_to_true_smf_bin_i
from .utils import lupton_log10


SHAMNET11_dim_in, SHAMNET11_dim_out = 5, 1
LOGSM_PRED_MIN = -5.0
LOGSM_PRED_MAX = 25


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
SHAMNET_DRN = os.path.join(_THIS_DRNAME, "data")
SHAMNET_BN = "shamnet_64_32_8_4.31.h5"
SHAMNET_FN = os.path.join(SHAMNET_DRN, SHAMNET_BN)
SHAMNET_PARAMS, DENSE_LAYERS = load_network_from_h5(SHAMNET_FN)[2:4]

SHAMNetResult = namedtuple(
    "SHAMNetResult",
    [
        "net_init",
        "pred_logsm_from_logmh",
        "smhm_batch_loss",
        "smf_batch_loss",
        "smf_grads_batch_loss",
        "smf_and_grads_joint_loss",
        "pred_logsm_batch",
        "pred_smf_batch",
        "pred_lnsmf_grads_batch",
        "pred_logsm_grads",
        "pred_logsm_scat_and_derivs",
    ],
)


def get_sham_network(dense_layers=DENSE_LAYERS):
    """Function defines and returns the SHAMnet network functions."""
    return _get_init_shamnet(
        dense_layers,
        SHAMNET11_dim_in,
        SHAMNET11_dim_out,
        X_MINS,
        X_MAXS,
    )


def _get_init_shamnet(dense_sizes, dim_in, dim_out, x_mins, x_maxs):
    _net_init, net_pred = get_network("Selu", dim_in, dim_out, dense_sizes)

    def net_init(dense_sizes=DENSE_LAYERS, ran_key=None):
        if ran_key is None:
            msg = "Must use default SHAMNet params if not passing ran_key to net_init"
            assert dense_sizes is DENSE_LAYERS, msg
            return SHAMNET_PARAMS
        else:
            _net_params = _net_init(ran_key, (-1, dim_in))[1]
            return _net_params

    @jjit
    def _predict_logsm_from_X(net_params, X):
        scaled_Y = net_pred(net_params, X)
        logsm = _output_layer_sigmoid(scaled_Y).flatten()
        return logsm

    @jjit
    def _predict_logsm_from_logmh(net_params, a, b, s_lo, s_hi, logmh):
        smf_params = jnp.array((a, b))
        scatter_params = jnp.array((s_lo, s_hi))
        X = _pack_Xtrain(smf_params, scatter_params, logmh, x_mins, x_maxs)
        return _predict_logsm_from_X(net_params, X)

    @jjit
    def initializer_singlepoint_loss(net_params, loss_data):
        X, Y_target = loss_data
        Y_pred = _predict_logsm_from_X(net_params, X)
        return _mse(Y_pred, Y_target)

    _batch_mse_loss_initializer = jjit(
        vmap(initializer_singlepoint_loss, in_axes=(None, 0))
    )

    @jjit
    def smhm_batch_loss(net_params, loss_data_batch):
        return jnp.mean(_batch_mse_loss_initializer(net_params, loss_data_batch))

    @jjit
    def _predict_logsm_from_logmh_batch(net_params, a, b, s_lo, s_hi, logmh):
        return _predict_logsm_from_logmh(net_params, a, b, s_lo, s_hi, logmh)

    _pred_logsm_batch = jjit(
        vmap(
            _predict_logsm_from_logmh_batch,
            in_axes=(None, 0, 0, 0, 0, 0),
        )
    )

    @jjit
    def pred_logsm_batch(net_params, smf_param_batch, scatter_param_batch, logmh_batch):
        a = smf_param_batch[:, 0]
        b = smf_param_batch[:, 1]
        s_lo = scatter_param_batch[:, 0]
        s_hi = scatter_param_batch[:, 1]
        return _pred_logsm_batch(net_params, a, b, s_lo, s_hi, logmh_batch)

    @jjit
    def _shamnet_logsm_from_logmh_scalar(net_params, a, b, s_lo, s_hi, logmh):
        return _predict_logsm_from_logmh(net_params, a, b, s_lo, s_hi, logmh)[0]

    _dlogsm_dlogmh_jacobian = vmap(
        grad(_shamnet_logsm_from_logmh_scalar, argnums=5),
        in_axes=(None, None, None, None, None, 0),
    )

    _dlogsm_dparams = grad(_shamnet_logsm_from_logmh_scalar, argnums=(1, 2, 3, 4))
    pred_logsm_grads = jjit(vmap(_dlogsm_dparams, in_axes=(*[None] * 5, 0)))

    _val_dlogsm_dparams = value_and_grad(
        _shamnet_logsm_from_logmh_scalar, argnums=(1, 2, 3, 4)
    )
    pred_val_logsm_grads = vmap(_val_dlogsm_dparams, in_axes=(*[None] * 5, 0))

    def _shamnet_scatter_scalar(net_params, a, b, s_lo, s_hi, logmh):
        return _scatter_model(logmh, s_lo, s_hi)

    _dscat_dparams = value_and_grad(_shamnet_scatter_scalar, argnums=(1, 2, 3, 4))
    pred_val_scatter_grads = vmap(_dscat_dparams, in_axes=(*[None] * 5, 0))

    @jjit
    def shamnet_logsm_scat_and_derivs(
        shamnet_params, logphi, alpha, scatter_lo, scatter_hi, logmp_arr
    ):
        args = (shamnet_params, logphi, alpha, scatter_lo, scatter_hi, logmp_arr)
        logsm, logsm_grad = pred_val_logsm_grads(*args)
        logsm_grad = jnp.vstack(logsm_grad)

        scat, scat_grad = pred_val_scatter_grads(*args)
        scat_grad = jnp.vstack(scat_grad)
        return logsm, logsm_grad, scat, scat_grad

    @jjit
    def dlogsm_dlogmh_jacobian(net_params, smf_params, scatter_params, logmh_table):
        return _dlogsm_dlogmh_jacobian(
            net_params, *smf_params, *scatter_params, logmh_table
        )

    @jjit
    def _smf_prediction_no_scatter(net_params, X):
        logsm_table = _predict_logsm_from_X(net_params, X)
        dlogsm = jnp.diff(logsm_table)

        # Retrieve physical arrays from X
        _params = _unpack_Xtrain(X, x_mins, x_maxs)
        smf_params, scatter_params, logmh_table = _params

        # Compute cumulative abundance of input halos from the model
        shmf_params = _get_shmf_params_at_z(FIXED_REDSHIFT)
        log_cnd_halos = _log10_cumulative_shmf(logmh_table, *shmf_params)
        nd_halos = 10 ** _calculate_lgnd_from_lgcnd(log_cnd_halos, logmh_table)

        # Convert dn/dx to dn/dy
        logmh_table_mids = 0.5 * (logmh_table[:-1] + logmh_table[1:])
        dy_dx = dlogsm_dlogmh_jacobian(
            net_params, smf_params, scatter_params, logmh_table_mids
        )
        nd_galaxies = nd_halos / dy_dx

        return logsm_table, nd_galaxies, dlogsm

    @jjit
    def _shamnet_smf_prediction_from_X(net_params, smf_logsm_bins, X):
        _res = _smf_prediction_no_scatter(net_params, X)
        logsm_table, smf_table, dlogsm = _res

        # Retrieve physical arrays from X
        _params = _unpack_Xtrain(X, x_mins, x_maxs)
        smf_params, scatter_params, logmh_table = _params
        logmh_table_mids = 0.5 * (logmh_table[:-1] + logmh_table[1:])
        scatter = _scatter_model(logmh_table_mids, *scatter_params)

        smf_pred = add_scatter_to_true_smf(
            logsm_table, smf_table, scatter, smf_logsm_bins
        )
        logsm_pred = 0.5 * (smf_logsm_bins[:-1] + smf_logsm_bins[1:])
        return logsm_pred, smf_pred

    @jjit
    def predict_smf_singlepoint(
        net_params, smf_logsm_bins, smf_params, scatter_params, logmh
    ):
        X = _pack_Xtrain(smf_params, scatter_params, logmh, x_mins, x_maxs)
        return _shamnet_smf_prediction_from_X(net_params, smf_logsm_bins, X)

    predict_smf_batch = jjit(
        vmap(
            predict_smf_singlepoint,
            in_axes=(None, None, 0, 0, 0),
            out_axes=(None, 0),
        )
    )

    @jjit
    def _shamnet_smf_pred_bin_i_from_X(net_params, smf_bin_lo, smf_bin_hi, X):
        _res = _smf_prediction_no_scatter(net_params, X)
        logsm_table, smf_table, dlogsm = _res
        logsm_table_mids = 0.5 * (logsm_table[:-1] + logsm_table[1:])

        # Retrieve physical arrays from X
        _params = _unpack_Xtrain(X, x_mins, x_maxs)
        smf_params, scatter_params, logmh_table = _params
        logmh_table_mids = 0.5 * (logmh_table[:-1] + logmh_table[1:])
        scatter = _scatter_model(logmh_table_mids, *scatter_params)

        smf_pred_bin_i = _add_scatter_to_true_smf_bin_i(
            smf_table, smf_bin_lo, smf_bin_hi, logsm_table_mids, dlogsm, scatter
        )
        return smf_pred_bin_i

    @jjit
    def predict_smf_bin_i(
        net_params,
        smf_bin_lo,
        smf_bin_hi,
        smf_params,
        scatter_params,
        logmh,
        log10_clip,
    ):
        X = _pack_Xtrain(smf_params, scatter_params, logmh, x_mins, x_maxs)
        smf_i = _shamnet_smf_pred_bin_i_from_X(net_params, smf_bin_lo, smf_bin_hi, X)
        return lupton_log10(smf_i, log10_clip)

    _vv = (None, 0, 0, None, None, None, None)
    _pred_lnsmf_grads_bin_i_vmap = jjit(
        vmap(grad(predict_smf_bin_i, argnums=3), in_axes=_vv)
    )

    @jjit
    def _pred_lnsmf_grads(
        net_params, smf_logsm_bins, smf_params, scatter_params, logmh, log10_clip
    ):
        smf_bins_lo = smf_logsm_bins[:-1]
        smf_bins_hi = smf_logsm_bins[1:]
        args = (
            net_params,
            smf_bins_lo,
            smf_bins_hi,
            smf_params,
            scatter_params,
            logmh,
            log10_clip,
        )
        return _pred_lnsmf_grads_bin_i_vmap(*args)

    predict_lnsmf_grads_batch = jjit(
        vmap(
            _pred_lnsmf_grads,
            in_axes=(None, None, 0, 0, 0, None),
        )
    )

    @jjit
    def smf_batch_loss(net_params, smf_loss_data):
        (
            smf_tbatch,
            scatter_tbatch,
            logmh_tbatch,
            smf_logsm_bins,
            log10_clip,
        ) = smf_loss_data
        logsm_pred, smf_preds = predict_smf_batch(
            net_params,
            smf_logsm_bins,
            smf_tbatch,
            scatter_tbatch,
            logmh_tbatch,
        )
        smf_targets = predict_smf_from_smf_param_batch(logsm_pred, smf_tbatch)
        log_smf_preds = lupton_log10(smf_preds, log10_clip)
        log_smf_targets = lupton_log10(smf_targets, log10_clip)
        log_smf_diff_matrix = log_smf_preds - log_smf_targets
        return jnp.mean(log_smf_diff_matrix * log_smf_diff_matrix, axis=(1, 0))

    @jjit
    def smf_grads_batch_loss(net_params, loss_data):
        smf_batch, scatter_batch, logmh_batch, smf_logsm_bins, log10_clip = loss_data
        smf_logsm_binmids = 0.5 * (smf_logsm_bins[:-1] + smf_logsm_bins[1:])
        grads_target = predict_ln_smf_grads_from_params(
            smf_logsm_binmids, smf_batch, log10_clip
        )
        args = smf_logsm_bins, smf_batch, scatter_batch, logmh_batch, log10_clip
        grads_pred = predict_lnsmf_grads_batch(net_params, *args)
        return _mse(grads_pred, grads_target)

    @jjit
    def smf_and_grads_joint_loss(net_params, loss_data):
        smf_loss = smf_batch_loss(net_params, loss_data)
        smf_grads_loss = smf_grads_batch_loss(net_params, loss_data)
        return smf_loss + smf_grads_loss

    return SHAMNetResult(
        net_init,
        _predict_logsm_from_logmh,
        smhm_batch_loss,
        smf_batch_loss,
        smf_grads_batch_loss,
        smf_and_grads_joint_loss,
        pred_logsm_batch,
        predict_smf_batch,
        predict_lnsmf_grads_batch,
        pred_logsm_grads,
        shamnet_logsm_scat_and_derivs,
    )


@jjit
def _output_layer_sigmoid(x, x0=0.0, k=0.1, ymin=LOGSM_PRED_MIN, ymax=LOGSM_PRED_MAX):
    return _sigmoid(x, x0, k, ymin, ymax)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _pack_Xtrain(smf_params, scatter_params, logmh_table, x_mins, x_maxs):
    n_mh = logmh_table.size
    dim_in = smf_params.size + scatter_params.size + 1
    X = jnp.zeros((n_mh, dim_in))
    X = jops.index_update(X, jops.index[:, 0], smf_params[0])
    X = jops.index_update(X, jops.index[:, 1], smf_params[1])
    X = jops.index_update(X, jops.index[:, 2], scatter_params[0])
    X = jops.index_update(X, jops.index[:, 3], scatter_params[1])
    X = jops.index_update(X, jops.index[:, 4], logmh_table)
    return _unit_scale_traindata(X, x_mins, x_maxs)


@jjit
def _unpack_Xtrain(X, x_mins, x_maxs):
    unscaled_X = _unscale_traindata(X, x_mins, x_maxs)
    smf_params = unscaled_X[0, 0:2]
    scatter_params = unscaled_X[0, 2:4]
    logmh_table = unscaled_X[:, 4]
    return smf_params, scatter_params, logmh_table


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)
