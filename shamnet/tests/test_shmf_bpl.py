"""
"""
import numpy as np
from ..shmf_bpl import log10_cumulative_shmf_bpl


def test1():
    lgmp = np.linspace(8, 17, 5000)
    lg_cumu_nd = log10_cumulative_shmf_bpl(lgmp, 0)
    assert np.all(np.isfinite(lg_cumu_nd))
