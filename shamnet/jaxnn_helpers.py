"""Convenience functions wrapping the behavior of jax.stax"""
import h5py
import numpy as np
from jax.experimental import stax
from jax import numpy as jnp
from jax import jit as jjit
from copy import deepcopy
from jax import value_and_grad
from jax import random as jran

BIAS_OUTPAT = "biases_layer_{}"
WEIGHT_OUTPAT = "weights_layer_{}"


def get_network(act_func_name, dim_in, dim_out, dense_sizes):
    """Build a simply connected neural network with jax.stax

    Parameters
    ----------
    act_func_name : string
        Name of the activation function.
        Should be acceptable stax options such as 'Selu', 'Relu', 'Gelu', etc.
        Passing the bare functions names (e.g., 'selu', 'relu', 'gelu') will not work.

    dim_in : int
        Integer storing the dimension of the input layer

    dim_out : int
        Integer storing the dimension of the output layer

    dense_sizes : sequence
        Sequence storing the dimension of each dense hidden layer

    Returns
    -------
    net_init : network initializer

    net_pred : network apply function

    """
    act_func = _get_activation_func(act_func_name)
    seq = list(_network_generator(act_func, dim_in, dim_out, dense_sizes))
    net_init, net_pred = stax.serial(*seq)
    return net_init, net_pred


def write_trained_network_to_h5(
    outname, net_params, xmin, xmax, ymin, ymax, activation, loss_history
):
    """Store the input network as an hdf5 file

    Parameters
    ----------
    outname : string
        Name of the output file

    net_params : sequence
        jax.stax network weights and biases

    xmin : ndarray of shape (dim_in, )

    xmax : ndarray of shape (dim_in, )

    ymin : ndarray of shape (dim_out, )

    ymax : ndarray of shape (dim_out, )

    activation : string

    """
    bias_collector = [layer[0] for layer in net_params[0::2]]
    weight_collector = [layer[1] for layer in net_params[0::2]]
    n_dense_layers = len(bias_collector)

    with h5py.File(outname, "w") as hdf:
        for i in range(n_dense_layers):
            hdf[BIAS_OUTPAT.format(i)] = bias_collector[i]
            hdf[WEIGHT_OUTPAT.format(i)] = weight_collector[i]
        hdf["xmin"] = xmin
        hdf["xmax"] = xmax
        hdf["ymin"] = ymin
        hdf["ymax"] = ymax
        hdf["n_dense_layers"] = np.atleast_1d(n_dense_layers)
        s = "activation/" + activation  # Store activation function name as metadata
        hdf[s] = np.zeros(1)


def load_network_from_h5(fname):
    """Load the jax.stax network from the input hdf5 file

    Parameters
    ----------
    fname : string

    Returns
    -------
    net_init : function
        Network initializer

    net_pred : function
        Network apply function

    net_params : sequence
        Weights and biases of the trained network

    bounds : sequence
        (xmin, xmax, ymin, ymax) stores unit-scaling of input and target data

    """
    dense_params, xmin, xmax, ymin, ymax, act_name = _read_network_data_from_h5(fname)

    dim_in, dense_sizes, dim_out = _get_network_layer_sizes(dense_params)
    net_init, net_pred = get_network(act_name, dim_in, dim_out, dense_sizes)

    net_params = []
    for param in dense_params:
        net_params.append(param)
        net_params.append(tuple(()))  # Add empty tuple for each activation layer

    layers = (dim_in, *dense_sizes, dim_out)
    bounds = xmin, xmax, ymin, ymax
    return net_init, net_pred, net_params[:-1], layers, bounds


def data_stream(batch_size, *X, fixed_data=tuple()):
    """Yield an infinite stream of mini-batches of training data."""
    n_train = X[0].shape[0]
    n_complete_batches, leftover = divmod(n_train, batch_size)
    n_batches = n_complete_batches + bool(leftover)

    while True:
        for ibatch in range(n_batches):
            ifirst, ilast = ibatch * batch_size, (ibatch + 1) * batch_size
            batch_data = list((x[slice(ifirst, ilast)] for x in X))
            for x in fixed_data:
                batch_data.append(x)
            yield tuple(batch_data)


def _read_network_data_from_h5(fname):
    """Read the network stored by the write_network_to_h5 function"""
    bias_accumulator = []
    weight_accumulator = []
    with h5py.File(fname, "r") as hdf:
        n_dense_layers = hdf["n_dense_layers"][...][0]
        activation = list(hdf["activation"].keys())[0]  # Extract activation function
        for i in range(n_dense_layers):
            bias_accumulator.append(hdf[BIAS_OUTPAT.format(i)][...])
            weight_accumulator.append(hdf[WEIGHT_OUTPAT.format(i)][...])
        xmin = hdf["xmin"][...]
        xmax = hdf["xmax"][...]
        ymin = hdf["ymin"][...]
        ymax = hdf["ymax"][...]

    dense_params = [(b, w) for b, w in zip(bias_accumulator, weight_accumulator)]

    return dense_params, xmin, xmax, ymin, ymax, activation


def _get_network_layer_sizes(net_params):
    """Infer the architecture of the simply connected network from the parameters"""
    dense_params = [p for p in net_params if len(p) > 0]
    all_dense_layer_sizes = [p[1].size for p in dense_params]
    dim_in = all_dense_layer_sizes[0]
    dim_out = all_dense_layer_sizes[-1]
    latent_sizes = all_dense_layer_sizes[1:-1]
    return dim_in, latent_sizes, dim_out


def _network_generator(act_func, dim_in, dim_out, dense_sizes):
    """Generate the sequence passed to stax.serial for a simply connected networks"""
    yield stax.Dense(dim_in)
    yield act_func

    for size in dense_sizes:
        yield stax.Dense(size)
        yield act_func

    yield stax.Dense(dim_out)


def _get_activation_func(act_layer_name):
    """Return the jax.stax activation function layer corresonding to the input string

    Parameters
    ----------
    act_layer_name : string
        Name of the activation layer, e.g., Selu, Relu
        Should not be, e.g., selu, relu, as these refer to the bare functions
        rather than the layer names

    Returns
    -------
    res : jax.stax object passed as a layer to stax.serial

    """
    return getattr(stax, act_layer_name)


@jjit
def _unit_scale_traindata(X, xmins, xmaxs):
    """If xmax > xmin, unit-scale the training data, else do nothing

    Parameters
    ----------
    x : ndarray of shape (m, n)

    xmins : ndarray of shape (n, )

    xmaxs : ndarray of shape (n, )

    Returns
    -------
    result : ndarray of shape (m, n)

    Notes
    -----
    Training data must fit inside a rectangular box aligned with each dimension

    """
    X = jnp.atleast_2d(X)
    xmins = jnp.atleast_1d(xmins)
    xmaxs = jnp.atleast_1d(xmaxs)
    msk = xmins == xmaxs
    norm = jnp.where(msk, 1.0, xmaxs - xmins)
    offset = jnp.where(msk, 0.0, xmins)
    return (X - offset) / norm


@jjit
def _unscale_traindata(X, xmins, xmaxs):
    """Inverse function to _unit_scale_traindata"""
    X = jnp.atleast_2d(X)
    xmins = jnp.atleast_1d(xmins)
    xmaxs = jnp.atleast_1d(xmaxs)
    return jnp.where(xmins == xmaxs, X, xmins + (X * (xmaxs - xmins)))


def train_with_multiple_steps_per_batch(
    n_steps_per_minibatch,
    opt_func,
    get_pars,
    state,
    opt_data,
    update,
    loss_min=float("inf"),
    p_best=None,
):
    """Wrapper function used to train based on multiple steps per mini-batch."""
    if p_best is None:
        p_best = deepcopy(get_pars(state))

    loss_history = []
    for istep in range(n_steps_per_minibatch):
        pars = get_pars(state)
        loss, grads = value_and_grad(opt_func, argnums=0)(pars, opt_data)
        loss_history.append(float(loss))

        if loss < loss_min:
            p_best = deepcopy(pars)
            loss_min = loss

        state = update(istep, grads, state)
    return state, loss_history, p_best, loss_min


def train_with_single_steps_per_batch(
    n_steps_tot,
    loss_data_generator,
    opt_func,
    get_pars,
    state,
    update,
    loss_min=float("inf"),
    p_best=None,
    starting_step=0,
):
    """Wrapper function used to train based on multiple steps per mini-batch."""
    if p_best is None:
        p_best = deepcopy(get_pars(state))

    loss_history = []
    for istep in range(n_steps_tot):
        opt_data = next(loss_data_generator)
        pars = get_pars(state)
        loss, grads = value_and_grad(opt_func, argnums=0)(pars, opt_data)
        loss_history.append(float(loss))

        if loss < loss_min:
            p_best = deepcopy(pars)
            loss_min = loss

        gd_step = starting_step + istep
        state = update(gd_step, grads, state)
    return state, gd_step, loss_history, p_best, loss_min


def get_randomly_spaced_array(ran_key, n, lo, hi):
    """Retrieve a 1d grid with randomly spaced points. Endpoints are held fixed,
    and intermediary points will be randomly staggered by +- dlogm/2.

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    n : int
        Size of the array

    lo : float

    hi : float

    Returns
    -------
    xarr : ndarray of shape (n, )

    """
    __, key = jran.split(ran_key)
    xarr = np.linspace(lo, hi, n)
    dx = np.diff(xarr)[0]
    u = jran.uniform(key, minval=-dx / 2, maxval=dx / 2, shape=(n - 2,))
    xarr[1:-1] = np.array(u) + xarr[1:-1]
    return xarr
