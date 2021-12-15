"""Functions calculating differentiable summary statistics with Numba."""
import os
import multiprocessing

import Corrfunc
import numpy as np
from numba import njit, vectorize


__all__ = (
    "triweighted_kernel_histogram",
    "triweighted_kernel_histogram_with_derivs",
    "triweighted_kernel_deltasigma_with_derivs",
    "triweighted_kernel_wprp_with_derivs",
)


def triweighted_kernel_deltasigma_with_derivs(
    log10mstar,
    log10mstar_jac,
    sigma,
    sigma_jac,
    ds_per_object,
    bins,
):
    """Calculate a triweight-kernel weighted histogram of log10(M*).

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos,)
    log10mstar_jac : ndarray, shape (n_params, n_halos)
    sigma : ndarray, shape (n_halos,)
    sigma_jac : ndarray, shape (n_params, n_halos)
    ds_per_object : ndarray, shape (n_halos, n_ds)
    bins : ndarray, shape (n_bin_edges,)

    Returns
    -------
    ds : ndarray, shape (n_bin_edges-1, n_ds)
    ds_jac : ndarray, shape (n_bin_edges-1, n_ds, n_params)
    """
    _ds, _ds_grad, nrms, jac_nrm = _numba_triweighted_kernel_deltasigma_with_derivs(
        np.atleast_1d(np.array(log10mstar)).astype("f8"),
        np.atleast_2d(np.array(log10mstar_jac)).astype("f8"),
        np.atleast_1d(np.array(sigma)).astype("f8"),
        np.atleast_2d(np.array(sigma_jac)).astype("f8"),
        np.atleast_2d(np.array(ds_per_object)).astype("f8"),
        np.atleast_1d(np.array(bins)).astype("f8"),
    )
    ds_grad = _ds_grad / nrms.reshape(-1, 1, 1) - (
        _ds.reshape(_ds.shape[0], _ds.shape[1], 1)
        / nrms.reshape(-1, 1, 1) ** 2
        * jac_nrm.reshape(jac_nrm.shape[0], 1, jac_nrm.shape[1])
    )
    ds = _ds / nrms.reshape(-1, 1)
    return ds, ds_grad


@njit
def _numba_triweighted_kernel_deltasigma_with_derivs(
    log10mstar,
    log10mstar_jac,
    sigma,
    sigma_jac,
    ds_per_object,
    bins,
):
    """Calculate a triweight-kernel weighted histogram of log10(M*).

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos,)
    log10mstar_jac : ndarray, shape (n_params, n_halos)
    sigma : ndarray, shape (n_halos,)
    sigma_jac : ndarray, shape (n_params, n_halos)
    ds_per_object : ndarray, shape (n_halos, n_ds)
    bins : ndarray, shape (n_bin_edges,)

    Returns
    -------
    ds : ndarray, shape (n_bin_edges-1, n_ds)
    ds_jac : ndarray, shape (n_bin_edges-1, n_ds, n_params)
    """
    n_bins = bins.shape[0] - 1
    n_params = log10mstar_jac.shape[0]
    n_data = log10mstar.shape[0]
    n_ds = ds_per_object.shape[1]

    hist = np.zeros((n_bins, n_ds), dtype=np.float64)
    hist_jac = np.zeros((n_bins, n_ds, n_params), dtype=np.float64)
    jac_sum = np.zeros((n_bins, n_params), dtype=np.float64)
    nrms = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_data):
        x = log10mstar[i]
        sig = sigma[i]
        last_cdf = _tw_cuml_kern(bins[0], x, sig)
        last_cdf_deriv = _tw_kern(bins[0], x, sig)

        for j in range(n_bins):
            new_cdf = _tw_cuml_kern(bins[j + 1], x, sig)
            new_cdf_deriv = _tw_kern(bins[j + 1], x, sig)

            # get the hist weight
            weight = new_cdf - last_cdf
            nrms[j] += weight
            for n in range(n_ds):
                hist[j, n] += weight * ds_per_object[i, n]

            # do the derivs
            for k in range(n_params):
                fac1 = log10mstar_jac[k, i] - x / sig * sigma_jac[k, i]
                fac2 = sigma_jac[k, i] / sig
                jac_term = last_cdf_deriv * (fac1 + bins[j] * fac2) - new_cdf_deriv * (
                    fac1 + bins[j + 1] * fac2
                )
                jac_sum[j, k] += jac_term
                for n in range(n_ds):
                    hist_jac[j, n, k] += jac_term * ds_per_object[i, n]

            last_cdf = new_cdf
            last_cdf_deriv = new_cdf_deriv

    return hist, hist_jac, nrms, jac_sum


def triweighted_kernel_histogram_with_derivs(
    log10mstar, log10mstar_jac, sigma, sigma_jac, bins
):
    """Calculate a triweight-kernel weighted histogram of log10(M*).

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos,)
    log10mstar_jac : ndarray, shape (n_params, n_halos)
    sigma : ndarray, shape (n_halos,)
    sigma_jac : ndarray, shape (n_params, n_halos)
    bins : ndarray, shape (n_bin_edges,)

    Returns
    -------
    hist : ndarray, shape (n_bin_edges-1,)
    hist_mean_jac : ndarray, shape (n_bin_edges-1, n_params)
    hist_sigma_jac : ndarray, shape (n_bin_edges-1, n_params)
    """
    return _numba_triweighted_kernel_histogram_with_derivs(
        np.atleast_1d(np.array(log10mstar)).astype("f8"),
        np.atleast_2d(np.array(log10mstar_jac)).astype("f8"),
        np.atleast_1d(np.array(sigma)).astype("f8"),
        np.atleast_2d(np.array(sigma_jac)).astype("f8"),
        np.atleast_1d(np.array(bins)).astype("f8"),
    )


@njit
def _numba_triweighted_kernel_histogram_with_derivs(
    log10mstar, log10mstar_jac, sigma, sigma_jac, bins
):
    """Calculate a triweight-kernel weighted histogram of log10(M*).

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos,)
    log10mstar_jac : ndarray, shape (n_params, n_halos)
    sigma : ndarray, shape (n_halos,)
    sigma_jac : ndarray, shape (n_params, n_halos)
    bins : ndarray, shape (n_bin_edges,)

    Returns
    -------
    hist : ndarray, shape (n_bin_edges-1,)
    hist_mean_jac : ndarray, shape (n_bin_edges-1, n_params)
    hist_sigma_jac : ndarray, shape (n_bin_edges-1, n_params)
    """
    n_bins = bins.shape[0] - 1
    n_params = log10mstar_jac.shape[0]
    n_data = log10mstar.shape[0]

    hist = np.zeros(n_bins, dtype=np.float64)
    hist_jac = np.zeros((n_bins, n_params), dtype=np.float64)

    for i in range(n_data):
        x = log10mstar[i]
        sig = sigma[i]
        last_cdf = _tw_cuml_kern(bins[0], x, sig)
        last_cdf_deriv = _tw_kern(bins[0], x, sig)

        for j in range(n_bins):
            new_cdf = _tw_cuml_kern(bins[j + 1], x, sig)
            new_cdf_deriv = _tw_kern(bins[j + 1], x, sig)

            # get the hist weight
            weight = new_cdf - last_cdf
            hist[j] += weight

            # do the derivs
            for k in range(n_params):
                fac1 = log10mstar_jac[k, i] - x / sig * sigma_jac[k, i]
                fac2 = sigma_jac[k, i] / sig
                hist_jac[j, k] += last_cdf_deriv * (
                    fac1 + bins[j] * fac2
                ) - new_cdf_deriv * (fac1 + bins[j + 1] * fac2)

            last_cdf = new_cdf
            last_cdf_deriv = new_cdf_deriv

    return hist, hist_jac


def triweighted_kernel_histogram(data, scatter, bins):
    """Sum the contribution of triweighted-kernel data to each bin.

    Parameters
    ----------
    data : ndarray of shape (nhalos,)
    scatter : float or ndarray of shape (nhalos,)
    bins : ndarray of shape (nbins,)

    Returns
    -------
    weighted_hist : ndarray of shape (nbins-1,)
    """
    data, scatter = _get_1d_arrays(data, scatter)
    bins = np.atleast_1d(bins).astype("f8")
    nbins = bins.shape[0]
    weighted_hist = np.zeros(nbins - 1).astype("f8")

    _numba_tw_hist(data, scatter, bins, weighted_hist)
    return weighted_hist


@vectorize
def _tw_cuml_kern(x, m, h):
    """CDF of the triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.
    m : array-like or scalar
        The mean of the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern_cdf : array-like or scalar
        The value of the kernel CDF.
    """
    y = (x - m) / h
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


@vectorize
def _tw_kern(x, m, h):
    """Triweight kernel.

    Parameters
    ----------
    x : array-like or scalar
        The value at which to evaluate the kernel.
    m : array-like or scalar
        The mean of the kernel.
    h : array-like or scalar
        The approximate 1-sigma width of the kernel.

    Returns
    -------
    kern : array-like or scalar
        The value of the kernel.
    """
    z = (x - m) / h
    if z < -3 or z > 3:
        return 0
    else:
        return 35 / 96 * (1 - (z / 3) ** 2) ** 3 / h


@njit
def _numba_tw_hist(data, scatter, bins, khist):
    """Numba kernel for the triweighted kernel histogram.

    Parameters
    ----------
    data : ndarray of shape (ndata,)
    scatter : ndarray of shape (ndata,)
    bins : ndarray of shape (nbins,)
    khist : ndarray of shape (nbins-1,)
        Empty array used to store the weighted histogram
    """
    ndata = len(data)
    nbins = len(bins)
    bot = bins[0]

    for i in range(ndata):
        x = data[i]
        scale = scatter[i]

        last_cdf = _tw_cuml_kern(x, bot, scale)
        for j in range(1, nbins):
            bin_edge = bins[j]
            new_cdf = _tw_cuml_kern(x, bin_edge, scale)
            weight = last_cdf - new_cdf
            khist[j - 1] += weight
            last_cdf = new_cdf


def _get_1d_arrays(*args):
    """Return a list of ndarrays of the same length."""
    results = [np.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)
    return [np.zeros(npts).astype(arr.dtype) + arr for arr in results]


def triweighted_kernel_wprp_with_derivs(
    log10mstar,
    log10mstar_jac,
    sigma,
    sigma_jac,
    x,
    y,
    z,
    zmax,
    boxsize,
    rp_bins,
    log10mstar_bins,
):
    """Calculate a triweight-kernel weighted histogram of log10(M*).

    Parameters
    ----------
    log10mstar : ndarray, shape (n_halos,)
    log10mstar_jac : ndarray, shape (n_params, n_halos)
    sigma : ndarray, shape (n_halos,)
    sigma_jac : ndarray, shape (n_params, n_halos)
    x, y, z : ndarray, shape (n_halos,)
        The positions in projected (x, y) and redshift-space (z).
    zmax : float
        The range over which to integrate the redhsift
        correlation function to get wp(rp).
    boxsize : float
        The length of the periodic box.
    rp_bins : ndarray, shape (n_rp_bins)
        The projected radius bin edges.
    log10mstar_bins : ndarray, shape (n_bin_edges,)
        The bins in stellar mass. wp(rp) and its derivatives are computed
        for each bin in stellar mass.

    Returns
    -------
    wprp : ndarray, shape (n_bin_edges-1, n_ds)
    wprp_jac : ndarray, shape (n_bin_edges-1, n_ds, n_params)
    """
    wprp = []
    wprp_jac = []

    rpbins_squared = rp_bins ** 2

    for bind in range(len(log10mstar_bins) - 1):
        w, w_jac = _compute_bin_weights_and_derivs(
            np.atleast_1d(np.array(log10mstar)).astype("f8"),
            np.atleast_2d(np.array(log10mstar_jac)).astype("f8"),
            np.atleast_1d(np.array(sigma)).astype("f8"),
            np.atleast_2d(np.array(sigma_jac)).astype("f8"),
            log10mstar_bins[bind],
            log10mstar_bins[bind + 1],
        )

        msk = w > 0
        _wprp, _wprp_jac = _wprp_weighted_with_derivs_serial_periodic_cpu_corrfunc(
            x1=x[msk],
            y1=y[msk],
            z1=z[msk],
            w1=w[msk],
            dw1=w_jac[:, msk],
            rpbins_squared=rpbins_squared,
            zmax=zmax,
            boxsize=boxsize,
        )
        wprp.append(_wprp)
        wprp_jac.append(_wprp_jac.T)

    return np.array(wprp), np.array(wprp_jac)


@njit
def _compute_bin_weights_and_derivs(
    log10mstar, log10mstar_jac, sigma, sigma_jac, log10mstar_low, log10mstar_hi
):
    n_params = log10mstar_jac.shape[0]
    n_data = log10mstar.shape[0]

    w = np.zeros(n_data, dtype=np.float64)
    w_jac = np.zeros((n_data, n_params), dtype=np.float64)

    for i in range(n_data):
        x = log10mstar[i]
        sig = sigma[i]
        last_cdf = _tw_cuml_kern(log10mstar_low, x, sig)
        last_cdf_deriv = _tw_kern(log10mstar_low, x, sig)

        new_cdf = _tw_cuml_kern(log10mstar_hi, x, sig)
        new_cdf_deriv = _tw_kern(log10mstar_hi, x, sig)

        w[i] = new_cdf - last_cdf

        # do the derivs
        for k in range(n_params):
            fac1 = log10mstar_jac[k, i] - x / sig * sigma_jac[k, i]
            fac2 = sigma_jac[k, i] / sig
            w_jac[i, k] = last_cdf_deriv * (
                fac1 + log10mstar_low * fac2
            ) - new_cdf_deriv * (fac1 + log10mstar_hi * fac2)

    return w, w_jac.T


def _compute_rr_rrgrad(w, dw, volratio):
    w_tot = np.sum(w)
    w2_tot = np.sum(w ** 2)
    n_eff = w_tot ** 2 / w2_tot
    dw_tot = np.sum(dw, axis=1)
    wdw_tot = np.sum(dw * w, axis=1)

    # finally get rr and drr
    return _compute_rr_rrgrad_eff(w_tot, dw_tot, wdw_tot, n_eff, volratio)


def _compute_rr_rrgrad_eff(w_tot, dw_tot, wdw_tot, n_eff, volratio):
    rr = w_tot ** 2 * (1 - 1.0 / n_eff) * volratio
    rr_grad = (
        2 * (w_tot * dw_tot.reshape(-1, 1, 1) - wdw_tot.reshape(-1, 1, 1)) * volratio
    )
    return rr, rr_grad


def _wprp_weighted_with_derivs_serial_periodic_cpu_corrfunc(
    *,
    x1,
    y1,
    z1,
    w1,
    dw1,
    rpbins_squared,
    zmax,
    boxsize,
):
    """Compute wp(rp) w/ derivs for a periodic volume.
    Parameters
    ----------
    x1, y1, z1, w1 : array-like, shape (n_pts,)
        The arrays of positions and weights for the first set of points.
    dw1 : array-lke, shape (n_grads, n_pts,)
        The array of weight gradients for the first set of points.
    rpbins_squared : array-like, shape (n_rpbins+1,)
        Array of the squared bin edges in the `rp` direction. Note that
        this array is one longer than the number of bins in `rp` direction.
    zmax : float
        The maximum separation in the `pi` (or typically `z`) direction. Output
        bins in `z` direction are unit width, so make sure this is a whole number.
    boxsize : float
        The size of the total periodic volume of the simulation.

    Returns
    -------
    wprp : array-like, shape (n_rpbins,)
        The projected correlation function.
    wprp_grad : array-like, shape (n_grads, n_rpbins)
        The gradients of the projected correlation function.
    """

    n_grads = dw1.shape[0]
    n_rp = rpbins_squared.shape[0] - 1
    n_pi = int(zmax)

    # dd
    res = Corrfunc.theory.DDrppi(
        True,
        os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count()),
        zmax,
        np.sqrt(rpbins_squared),
        x1,
        y1,
        z1,
        weights1=w1,
        periodic=True,
        boxsize=boxsize,
        weight_type="pair_product",
    )
    dd = (
        res["weightavg"].reshape((n_rp, n_pi)) * res["npairs"].reshape((n_rp, n_pi)) / 2
    )

    # now do the grad terms
    dd_grad = np.zeros((n_grads, n_rp, n_pi))
    for g in range(n_grads):
        res = Corrfunc.theory.DDrppi(
            False,
            os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count()),
            zmax,
            np.sqrt(rpbins_squared),
            x1,
            y1,
            z1,
            weights1=w1,
            X2=x1,
            Y2=y1,
            Z2=z1,
            weights2=dw1[g, :],
            periodic=True,
            boxsize=boxsize,
            weight_type="pair_product",
        )
        dd_grad[g, :, :] += (
            res["weightavg"].reshape((n_rp, n_pi))
            * res["npairs"].reshape((n_rp, n_pi))
            / 2
        )

        res = Corrfunc.theory.DDrppi(
            False,
            os.environ.get("OMP_NUM_THREADS", multiprocessing.cpu_count()),
            zmax,
            np.sqrt(rpbins_squared),
            x1,
            y1,
            z1,
            weights1=dw1[g, :],
            X2=x1,
            Y2=y1,
            Z2=z1,
            weights2=w1,
            periodic=True,
            boxsize=boxsize,
            weight_type="pair_product",
        )
        dd_grad[g, :, :] += (
            res["weightavg"].reshape((n_rp, n_pi))
            * res["npairs"].reshape((n_rp, n_pi))
            / 2
        )

    # now do norm by RR and compute proper grad

    # this is the volume of the shell
    dpi = 1.0  # here to make the code clear, always true
    volfac = np.pi * (rpbins_squared[1:] - rpbins_squared[:-1])
    volratio = volfac[:, None] * np.ones(n_pi) * dpi / boxsize ** 3

    # finally get rr and drr
    rr, rr_grad = _compute_rr_rrgrad(w1, dw1, volratio)

    # now produce value and derivs
    xirppi = dd / rr - 1
    xirppi_grad = (
        dd_grad / rr[None, :, :] - dd[None, :, :] / rr[None, :, :] ** 2 * rr_grad
    )

    # integrate over pi
    wprp = 2.0 * dpi * np.sum(xirppi, axis=-1)
    wprp_grad = 2.0 * dpi * np.sum(xirppi_grad, axis=-1)

    return wprp, wprp_grad
