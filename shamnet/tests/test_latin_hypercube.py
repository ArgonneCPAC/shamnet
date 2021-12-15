"""
"""
import numpy as np
from ..latin_hypercube import latin_hypercube, latin_hypercube_from_cov


def test_latin_hypercube1():
    param_bounds = [(-3, 2), (-2, 3), (0, 5)]
    npts = 5000
    lhs_box = latin_hypercube(param_bounds, npts)
    for idim in range(len(param_bounds)):
        assert np.all(lhs_box[:, idim] >= param_bounds[idim][0])
        assert np.all(lhs_box[:, idim] <= param_bounds[idim][1])


def test_latin_hypercube2():
    param_bounds = [(-3, 2), (-2, 3), (5, 0)]
    npts = 5000
    try:
        latin_hypercube(param_bounds, npts)
    except AssertionError:
        pass


def test_latin_hypercube_from_diagonal_cov():
    mu = np.array((4.0, -5.0))
    cov = np.array([[0.014, 0.0], [0.0, 0.015]])
    sig = 5
    n = 5000

    lhs = latin_hypercube_from_cov(mu, cov, sig, n)
    for idim in range(2):
        assert np.all(lhs[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))

    lhs2 = latin_hypercube_from_cov(mu, cov, (sig, sig), n)
    for idim in range(2):
        assert np.all(lhs2[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs2[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))


def test_latin_hypercube_from_non_diagonal_cov():
    mu = np.array((4.0, -5.0))
    cov = np.array([[0.014, 0.0075], [0.0075, 0.015]])
    sig = 5
    n = 5000

    lhs = latin_hypercube_from_cov(mu, cov, sig, n)
    for idim in range(2):
        assert ~np.all(lhs[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert ~np.all(lhs[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))
    for idim in range(2):
        assert np.all(lhs[:, idim] > mu[idim] - 2 * sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs[:, idim] < mu[idim] + 2 * sig * np.sqrt(cov[idim, idim]))

    lhs2 = latin_hypercube_from_cov(mu, cov, (sig, sig), n)
    for idim in range(2):
        assert ~np.all(lhs2[:, idim] > mu[idim] - sig * np.sqrt(cov[idim, idim]))
        assert ~np.all(lhs2[:, idim] < mu[idim] + sig * np.sqrt(cov[idim, idim]))
    for idim in range(2):
        assert np.all(lhs2[:, idim] > mu[idim] - 2 * sig * np.sqrt(cov[idim, idim]))
        assert np.all(lhs2[:, idim] < mu[idim] + 2 * sig * np.sqrt(cov[idim, idim]))
