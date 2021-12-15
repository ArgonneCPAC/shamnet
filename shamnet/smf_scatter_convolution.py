"""Module adds variable scatter to a true SMF to predict an observed SMF."""
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z ** 7 / 69984
        + 7 * z ** 5 / 2592
        - 35 * z ** 3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _get_galaxy_weight(logsm, scatter, logsm_low, logsm_high):
    """Triweight kernel integrated across the boundaries of a single bin."""
    a = _tw_cuml_kern(logsm, logsm_low, scatter)
    b = _tw_cuml_kern(logsm, logsm_high, scatter)
    return a - b


@jjit
def _add_scatter_to_true_smf_bin_i(
    true_smf_bin_i, bin_i_lo, bin_i_hi, logsm_table, dlogsm_table, scatter_table
):
    """Add scatter to the true SMF to compute the SMF that would be measured
    in a bin of observed stellar mass between bin_i_lo, bin_i_hi.

    For every tabulated point of the true SMF, integrate a triweight kernel centered at
    the point across the boundaries of the observed SMF bins. Repeat for every point
    and sum the results to calculate the total number of galaxies that would be
    measured to lie within the bin of observed stellar mass.

    """
    table_weights_bin_i = _get_galaxy_weight(
        logsm_table, scatter_table, bin_i_lo, bin_i_hi
    )
    bin_i_width = bin_i_hi - bin_i_lo
    bin_i_integrand = table_weights_bin_i * true_smf_bin_i * dlogsm_table
    smf_with_scatter_bin_i = jnp.sum(bin_i_integrand) / bin_i_width
    return smf_with_scatter_bin_i


_a = (None, 0, 0, None, None, None)
_add_scatter_to_true_smf = jjit(vmap(_add_scatter_to_true_smf_bin_i, in_axes=_a))


@jjit
def add_scatter_to_true_smf(
    true_smf_logsm_bins, true_smf, scatter, scatter_smf_logsm_bins
):
    """Include scatter in stellar mass in the prediction of the SMF.

    Parameters
    ----------
    true_smf_logsm_bins : ndarray of shape (n_bins_true, )
        Base-10 log of Mstar defining the bin edges used to define the true SMF

    true_smf : ndarray of shape (n_bins_true-1, )
        True SMF evaluted at the midpoints of true_smf_logsm_bins

    scatter : float or ndarray of shape (n_bins_true-1, )
        Value of scatter in dex for each point of the true SMF

    scatter_smf_logsm_bins : ndarray of shape (n_bins_out, )
        Base-10 log of Mstar defining the bin edges used to define the observed SMF

    Returns
    -------
    smf_with_scatter : ndarray of shape (n_bins_out-1, )
        Observed SMF evaluted at the midpoints of scatter_smf_logsm_bins

    """

    scatter_smf_logsm_bins_lo_bounds = scatter_smf_logsm_bins[:-1]
    scatter_smf_logsm_bins_hi_bounds = scatter_smf_logsm_bins[1:]
    true_smf_logsm_bin_mids = 0.5 * (true_smf_logsm_bins[:-1] + true_smf_logsm_bins[1:])
    true_smf_dlogsm = jnp.diff(true_smf_logsm_bins)
    return _add_scatter_to_true_smf(
        true_smf,
        scatter_smf_logsm_bins_lo_bounds,
        scatter_smf_logsm_bins_hi_bounds,
        true_smf_logsm_bin_mids,
        true_smf_dlogsm,
        scatter,
    )
