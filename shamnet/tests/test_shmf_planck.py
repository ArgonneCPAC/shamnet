"""
"""
import h5py
import os
import numpy as np
from ..shmf_planck import _log10_cumulative_shmf
from ..shmf_planck import DEFAULT_SHMF_PARAM_BOUNDS, DEFAULT_SHMF_PARAMS
from ..shmf_planck import Z0P5_PARAMS, Z1P0_PARAMS, Z2P0_PARAMS


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_default_shmf_agrees_with_bpl_at_z0():
    fname = os.path.join(_THIS_DRNAME, "testing_data", "bpl_shmf_z0.h5")
    with h5py.File(fname, "r") as hdf:
        logmp_table = hdf["logmp_table"][...]
        lognd_table = hdf["lognd_table"][...]
    lognd_pred = _log10_cumulative_shmf(logmp_table, *DEFAULT_SHMF_PARAMS.values())
    mse = np.mean((lognd_pred - lognd_table) ** 2)
    assert mse < 0.005


def test_default_shmf_agrees_with_bpl_at_z0p5():
    fname = os.path.join(_THIS_DRNAME, "testing_data", "bpl_shmf_z0p5.h5")
    with h5py.File(fname, "r") as hdf:
        logmp_table = hdf["logmp_table"][...]
        lognd_table = hdf["lognd_table"][...]
    lognd_pred = _log10_cumulative_shmf(logmp_table, *Z0P5_PARAMS.values())
    mse = np.mean((lognd_pred - lognd_table) ** 2)
    assert mse < 0.005


def test_default_shmf_agrees_with_bpl_at_z1p0():
    fname = os.path.join(_THIS_DRNAME, "testing_data", "bpl_shmf_z1p0.h5")
    with h5py.File(fname, "r") as hdf:
        logmp_table = hdf["logmp_table"][...]
        lognd_table = hdf["lognd_table"][...]
    lognd_pred = _log10_cumulative_shmf(logmp_table, *Z1P0_PARAMS.values())
    mse = np.mean((lognd_pred - lognd_table) ** 2)
    assert mse < 0.005


def test_default_shmf_agrees_with_bpl_at_z2p0():
    fname = os.path.join(_THIS_DRNAME, "testing_data", "bpl_shmf_z2p0.h5")
    with h5py.File(fname, "r") as hdf:
        logmp_table = hdf["logmp_table"][...]
        lognd_table = hdf["lognd_table"][...]
    lognd_pred = _log10_cumulative_shmf(logmp_table, *Z2P0_PARAMS.values())
    mse = np.mean((lognd_pred - lognd_table) ** 2)
    assert mse < 0.005


def test_default_shmf_params_are_in_bounds():
    for param_name, bounds in DEFAULT_SHMF_PARAM_BOUNDS.items():
        default_value = DEFAULT_SHMF_PARAMS[param_name]
        assert bounds[0] < default_value < bounds[1]


def test_highz_shmf_params_are_in_bounds():
    for param_name, bounds in DEFAULT_SHMF_PARAM_BOUNDS.items():
        assert bounds[0] < Z0P5_PARAMS[param_name] < bounds[1]
        assert bounds[0] < Z1P0_PARAMS[param_name] < bounds[1]
        assert bounds[0] < Z2P0_PARAMS[param_name] < bounds[1]
