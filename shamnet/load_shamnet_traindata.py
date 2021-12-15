"""Module used to load the training data from an hdf5 file
and return arrays formatted appropriately for network training."""
from collections import OrderedDict
import h5py
from copy import deepcopy
import numpy as np
from jax import numpy as jnp
from jax import random as jran
from .shamnet_traindata import DEFAULT_SMHM_PARAMS
from .shamnet_traindata import _logsm_from_logmh
from .jaxnn_helpers import _unit_scale_traindata, get_randomly_spaced_array
from .shmf_planck import _get_shmf_params_at_z
from .generate_shamnet_traindata import X_MINS, X_MAXS, FIXED_REDSHIFT

LOGMH_BOUNDS = X_MINS[-1], X_MAXS[-1]


def load_initializer_tdata_and_vdata(
    fn, n_train, ran_key, low_logmh, high_logmh, n_mh=200, n_validate=100, tol=-2.5
):
    """Load from disk the training and validation data for SHAMnet.

    Parameters
    ----------
    fn : string
        h5 filename storing the best-fit SMHM params corresponding to SMF params.

    ntrain : int
        Number of points of training data

    ran_key : jax PRNGKey
        Used to randomly select n_train points

    n_validate : int, optional
        Number of points of validation data

    tol : float, optional
        largest value of the loss accepted as a valid training point

    Returns
    -------
    tdata : sequence
        Return value of the _get_init_traindata_from_smhms function

    vdata : sequence
        Return value of the _get_init_traindata_from_smhms function

    """
    _res = load_shamnet11_initializer_traindata(fn)
    traindata_X, traindata_Y, traindata_loss = _res[0:3]
    smf_tdata, scatter_tdata, all_smhm_tdata, bounds_X, bounds_Y = _res[3:]
    fixed_shmf_params = _get_shmf_params_at_z(FIXED_REDSHIFT)

    msk = traindata_loss < 10 ** tol
    n_clean = msk.sum()
    n_select = n_train + n_validate
    if n_clean < n_select:
        msg = (
            "Requested n_train = {0} + n_validate = {1} = {2} training points,\n"
            "but only {3} total points pass the requested tolerance"
        )
        raise ValueError(msg.format(n_train, n_validate, n_select, n_clean))
    clean_smf_data = smf_tdata[msk]
    clean_scatter_data = scatter_tdata[msk]
    clean_smhm_data = all_smhm_tdata[msk]

    downsample_key, ran_key = jran.split(ran_key)
    n_clean = clean_smf_data.shape[0]
    _indx = np.arange(0, n_clean).astype("i8")
    indx = jran.choice(downsample_key, _indx, shape=(n_select,), replace=False)
    smf_traindata = clean_smf_data[indx, :]
    scatter_traindata = clean_scatter_data[indx, :]
    smhm_traindata = clean_smhm_data[indx, :]

    _data = _get_init_traindata_from_smhms(
        ran_key,
        smf_traindata,
        scatter_traindata,
        smhm_traindata,
        fixed_shmf_params,
        low_logmh,
        high_logmh,
        n_mh,
    )
    X_all, Y_all, smf_all, scatter_all, logmh_all, smhm_all, fixed_shmf_params = _data
    X_tdata, X_vdata = X_all[:-n_validate], X_all[-n_validate:]
    Y_tdata, Y_vdata = Y_all[:-n_validate], Y_all[-n_validate:]
    smf_tdata, smf_vdata = smf_all[:-n_validate], smf_all[-n_validate:]
    scatter_tdata, scatter_vdata = scatter_all[:-n_validate], scatter_all[-n_validate:]
    logmh_tdata, logmh_vdata = logmh_all[:-n_validate], logmh_all[-n_validate:]
    smhm_tdata, smhm_vdata = smhm_all[:-n_validate], smhm_all[-n_validate:]
    tdata = (
        X_tdata,
        Y_tdata,
        smf_tdata,
        scatter_tdata,
        logmh_tdata,
        smhm_tdata,
        fixed_shmf_params,
    )
    vdata = (
        X_vdata,
        Y_vdata,
        smf_vdata,
        scatter_vdata,
        logmh_vdata,
        smhm_vdata,
        fixed_shmf_params,
    )
    return tdata, vdata


def _get_init_traindata_from_smhms(
    ran_key,
    smf_param_batch,
    scatter_param_batch,
    smhm_param_batch,
    fixed_shmf_params,
    low_logmh,
    high_logmh,
    n_mh,
):
    X_tbatch = []
    Y_tbatch = []
    logmh_tbatch = []
    smf_tbatch = []
    scatter_tbatch = []
    smhm_tbatch = []
    smf_key, ran_key = jran.split(ran_key)

    n_batch = smf_param_batch.shape[0]
    for i in range(n_batch):
        smf_params_i = smf_param_batch[i, :]
        scatter_params_i = scatter_param_batch[i, :]
        smhm_params_i = smhm_param_batch[i, :]
        logmh_table_i = _get_random_logmh_table(ran_key, n_mh, low_logmh, high_logmh)
        args = (
            logmh_table_i,
            smf_params_i,
            scatter_params_i,
            smhm_params_i,
            X_MINS,
            X_MAXS,
        )
        X_i, Y_i = _get_initializer_traindata(*args)
        X_tbatch.append(X_i)
        Y_tbatch.append(Y_i)
        smf_tbatch.append(smf_params_i)
        scatter_tbatch.append(scatter_params_i)
        logmh_tbatch.append(logmh_table_i)
        smhm_tbatch.append(smhm_params_i)

    X_tbatch = jnp.array(X_tbatch)
    Y_tbatch = jnp.array(Y_tbatch)
    smf_tbatch = jnp.array(smf_tbatch)
    scatter_tbatch = jnp.array(scatter_tbatch)
    logmh_tbatch = jnp.array(logmh_tbatch)
    smhm_tbatch = jnp.array(smhm_tbatch)
    return (
        X_tbatch,
        Y_tbatch,
        smf_tbatch,
        scatter_tbatch,
        logmh_tbatch,
        smhm_tbatch,
        fixed_shmf_params,
    )


def _get_random_logmh_table(ran_key, n_mh, lo, hi):
    """Retrieve a random table of log halo mass. Endpoints will be held fixed to
    LOGMH_BOUNDS set at top of module.
    Intermediary points will be randomly staggered by +- dlogm/2.

    """
    ret_key, lgm_key = jran.split(ran_key)
    assert lo >= LOGMH_BOUNDS[0]
    assert hi <= LOGMH_BOUNDS[1]
    return get_randomly_spaced_array(ran_key, n_mh, lo, hi)


def _get_initializer_traindata(
    logmh_table, smf_params, scatter_params, smhm_params, xmins, xmaxs
):
    """Transform the unscaled input 1-d arrays into scaled matrices."""
    _traindata = _pack_traindata_matrix(smf_params, scatter_params, logmh_table)
    traindata_X = _unit_scale_traindata(_traindata, xmins, xmaxs)
    traindata_Y = _logsm_from_logmh(smhm_params, logmh_table)
    return traindata_X, traindata_Y


def _pack_traindata_matrix(smf_params, scatter_params, logmh_table):
    """Construct the training matrix X from the two input 1-d arrays."""
    n_t = len(logmh_table)
    x = jnp.array((*smf_params, *scatter_params))
    n_x = len(x)
    X = np.zeros((n_t, n_x + 1))
    X[:, :-1] = jnp.tile(x, n_t).reshape((n_t, n_x))
    X[:, -1] = logmh_table
    return X


def load_shamnet11_initializer_traindata(fn):
    """
    Load the SHAMnet training data from the hdf5 file


    Parameters
    ----------
    fn : string

    Returns
    -------
    traindata_X : ndarray of shape (n_train, dim_in)

    traindata_Y : ndarray of shape (n_train, dim_out)

    traindata_loss : ndarray of shape (n_train, )

    """

    with h5py.File(fn, "r") as hdf:
        traindata_X = hdf["traindata_X"][...]
        traindata_Y = hdf["traindata_Y"][...]
        traindata_loss = hdf["traindata_loss"][...]
        fixed_smhm_param_dict = OrderedDict()
        for subkey in hdf["fixed_smhm_params"].keys():
            composite_key = "fixed_smhm_params" + "/" + subkey
            fixed_smhm_param_dict[subkey] = float(hdf[composite_key][...])

        X_mins = hdf["X_mins"][...]
        X_maxs = hdf["X_maxs"][...]
        smhm_mins = hdf["smhm_params_mins"][...]
        smhm_maxs = hdf["smhm_params_maxs"][...]

    bounds_X = X_mins, X_maxs
    bounds_Y = smhm_mins, smhm_maxs

    all_smhm_tdata = _get_all_smhm_params(traindata_Y, fixed_smhm_param_dict)

    smf_tdata = traindata_X[:, :2]
    scatter_tdata = traindata_X[:, 2:]

    return (
        traindata_X,
        traindata_Y,
        traindata_loss,
        smf_tdata,
        scatter_tdata,
        all_smhm_tdata,
        bounds_X,
        bounds_Y,
    )


def _get_all_smhm_params(Y, fixed_smhm_param_dict):
    varied_smhm_params = deepcopy(DEFAULT_SMHM_PARAMS)
    for key in fixed_smhm_param_dict.keys():
        varied_smhm_params.pop(key)

    all_smhm_params = np.zeros(shape=(Y.shape[0], len(DEFAULT_SMHM_PARAMS)))
    counter = 0
    for varied_param_name in varied_smhm_params.keys():
        indx = list(DEFAULT_SMHM_PARAMS.keys()).index(varied_param_name)
        all_smhm_params[:, indx] = Y[:, counter]
        counter += 1

    for fixed_param_name, fixed_val in fixed_smhm_param_dict.items():
        indx = list(DEFAULT_SMHM_PARAMS.keys()).index(fixed_param_name)
        all_smhm_params[:, indx] = fixed_val
    return all_smhm_params
