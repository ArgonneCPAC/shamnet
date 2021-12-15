"""
"""
import numpy as np
from jax import jit as jjit
from jax import value_and_grad
from jax.experimental import optimizers as jax_opt
from jax import numpy as jnp


@jjit
def jax_np_interp(x, xt, yt, indx_hi):
    """JAX-friendly implementation of np.interp.
    This is a relic from before this was implemented in JAX.
    Requires indx_hi to be precomputed, e.g., using np.searchsorted.

    Parameters
    ----------
    x : ndarray of shape (n, )
        Abscissa values in the interpolation

    xt : ndarray of shape (k, )
        Lookup table for the abscissa

    yt : ndarray of shape (k, )
        Lookup table for the ordinates

    Returns
    -------
    y : ndarray of shape (n, )
        Result of linear interpolation

    """
    indx_lo = indx_hi - 1
    xt_lo = xt[indx_lo]
    xt_hi = xt[indx_hi]
    dx_tot = xt_hi - xt_lo
    yt_lo = yt[indx_lo]
    yt_hi = yt[indx_hi]
    dy_tot = yt_hi - yt_lo
    m = dy_tot / dx_tot
    y = yt_lo + m * (x - xt_lo)
    return y


def jax_adam_wrapper(
    loss_func,
    params_init,
    loss_data,
    n_step,
    n_warmup=0,
    step_size=0.01,
    warmup_n_step=50,
    warmup_step_size=None,
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    n_warmup : int, optional
        Number of warmup iterations. At the end of the warmup, the best-fit parameters
        are used as input parameters to the final burn. Default is zero.

    warmup_n_step : int, optional
        Number of Adam steps to take during warmup. Default is 50.

    warmup_step_size : float, optional
        Step size to use during warmup phase. Default is 5*step_size.

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01.

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    fit_terminates : int
        0 if NaN or inf is encountered by the fitter, causing termination before n_step
        1 for a fit that terminates with no such problems

    """
    if warmup_step_size is None:
        warmup_step_size = 5 * step_size

    p_init = np.copy(params_init)
    for i in range(n_warmup):
        p_init = _jax_adam_wrapper(
            loss_func, p_init, loss_data, warmup_n_step, step_size=warmup_step_size
        )[0]

    if np.all(np.isfinite(p_init)):
        p0 = p_init
    else:
        p0 = params_init

    _res = _jax_adam_wrapper(loss_func, p0, loss_data, n_step, step_size=step_size)
    if len(_res[2]) < n_step:
        fit_terminates = 0
    else:
        fit_terminates = 1
    return (*_res, fit_terminates)


def _jax_adam_wrapper(loss_func, params_init, loss_data, n_step, step_size=0.01):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    """
    loss_arr = np.zeros(n_step).astype("f4") - 1.0
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)
    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    for istep in range(n_step):
        p = np.array(get_params(opt_state))

        loss, grads = value_and_grad(loss_func, argnums=0)(p, loss_data)

        no_nan_params = np.all(np.isfinite(p))
        no_nan_loss = np.isfinite(loss)
        no_nan_grads = np.all(np.isfinite(grads))
        if ~no_nan_params | ~no_nan_loss | ~no_nan_grads:
            if istep > 0:
                indx_best = np.nanargmin(loss_arr[:istep])
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_arr[indx_best]
            else:
                best_fit_params = np.copy(p)
                best_fit_loss = 999.99
            return (
                best_fit_params,
                best_fit_loss,
                loss_arr[:istep],
                params_arr[:istep, :],
            )
        else:
            params_arr[istep, :] = p
            loss_arr[istep] = loss
            opt_state = opt_update(istep, grads, opt_state)

    indx_best = np.nanargmin(loss_arr)
    best_fit_params = params_arr[indx_best]
    loss = loss_arr[indx_best]

    return best_fit_params, loss, loss_arr, params_arr


@jjit
def lupton_log10(t, log10_clip, t0=0.0, M0=0.0, alpha=1 / jnp.log(10.0)):
    """Clipped base-10 log function.

    Parameters
    ----------
    t : ndarray of shape (n, )

    log10_clip : float
        Returned values of t larger than log10_clip will agree with log10(t).
        Values smaller than log10(t) will converge to 10**log10_clip.

    Returns
    -------
    lup : ndarray of shape (n, )

    """
    k = 10.0 ** log10_clip
    return M0 + alpha * (jnp.arcsinh((t - t0) / (2 * k)) + jnp.log(k))
