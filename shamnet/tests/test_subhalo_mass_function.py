"""
"""
import os
import h5py
import numpy as np
from ..subhalo_mass_function import log10_cumulative_shmf
from ..subhalo_mass_function import DEFAULT_SHMF_PARAMS, DEFAULT_SHMF_PARAM_BOUNDS

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_default_shmf_agrees_with_bpl():
    fname = os.path.join(_THIS_DRNAME, "testing_data", "bpl_shmf_z0.h5")
    with h5py.File(fname, "r") as hdf:
        logmp_table = hdf["logmp_table"][...]
        lognd_table = hdf["lognd_table"][...]
    lognd_pred = log10_cumulative_shmf(logmp_table)
    mse = np.mean((lognd_pred - lognd_table) ** 2)
    assert mse < 0.005


def test_default_shmf_params_are_in_bounds():
    for param_name, bounds in DEFAULT_SHMF_PARAM_BOUNDS.items():
        default_value = DEFAULT_SHMF_PARAMS[param_name]
        assert bounds[0] < default_value < bounds[1]
