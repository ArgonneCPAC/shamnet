"""
"""
import numpy as np
from ..utils import jax_np_interp
from ..utils import lupton_log10

SEED = 43


def test_jax_np_interp_agrees_with_numpy():
    xt = np.linspace(0, 1, 50)
    yt = 5 + 2 * (xt - 1)
    rng = np.random.RandomState(SEED)
    x = rng.uniform(0, 1, 150)
    indx = np.searchsorted(xt, x)
    result = jax_np_interp(x, xt, yt, indx)
    np_result = np.interp(x, xt, yt)
    assert np.allclose(result, np_result, rtol=1e-4)


def test_base10_lupton_log_is_correct():
    x = np.logspace(-15, 0, 5000)
    log10x = np.log10(x)

    clip = -10.0
    lupton_log10x = lupton_log10(x, clip)

    agree_msk = log10x > clip + 1
    assert np.allclose(log10x[agree_msk], lupton_log10x[agree_msk], atol=0.01)

    disagree_msk = log10x > clip - 1
    assert not np.allclose(log10x[disagree_msk], lupton_log10x[disagree_msk], atol=0.01)
