"""
"""
import numpy as np
from ..smf_scatter_convolution import add_scatter_to_true_smf
from ..leja19_smf import _double_schechter_fitter, DEFAULT_DBL_SCHECHTER_FITTER


def test_add_scatter_to_true_smf_accepts_variable_scatter():
    n_true_smf_bins = 200
    logsm_bins_true_smf = np.linspace(8, 12, n_true_smf_bins)
    logsm_bin_mids_true_smf = 0.5 * (logsm_bins_true_smf[:-1] + logsm_bins_true_smf[1:])
    smf_params = np.array(list(DEFAULT_DBL_SCHECHTER_FITTER.values()))
    smf_table = _double_schechter_fitter(logsm_bin_mids_true_smf, *smf_params)
    scatter = 0.25
    n_obs_smf_bins = 50
    logsm_bins_obs_smf = np.linspace(9, 11.5, n_obs_smf_bins)
    smf_with_scatter = add_scatter_to_true_smf(
        logsm_bins_true_smf, smf_table, scatter, logsm_bins_obs_smf
    )
    smf_with_scatter2 = add_scatter_to_true_smf(
        logsm_bins_true_smf,
        smf_table,
        scatter + np.zeros(n_true_smf_bins - 1),
        logsm_bins_obs_smf,
    )
    assert np.allclose(smf_with_scatter, smf_with_scatter2)
