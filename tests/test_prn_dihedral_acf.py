import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plotting'))
from prn_dihedral_acf import cosine_acf
from prn_dihedral_acf import fit_log_decay
from prn_dihedral_acf import fit_exponential_acf


def test_direct_exponential_fit():
    lag=np.arange(0,40.01,.01)
    c=.95+.05*np.exp(-lag/.08)
    fitted,mask,info=fit_exponential_acf(lag,c,.95,35)
    np.testing.assert_allclose(info['tau_ps'],.08,rtol=1e-6)
    np.testing.assert_allclose(fitted,c,atol=1e-8)
    assert not mask[lag>=35].any()
    c[100]=.94
    _,mask,_=fit_exponential_acf(lag,c,.95,35)
    assert mask[100]  # No log-domain filtering in the direct fit.


def test_free_plateau_fit():
    lag=np.arange(0,5.01,.01)
    c=.956+.044*np.exp(-lag/.07)
    _,_,info=fit_exponential_acf(lag,c,.951,5.0000001,fit_plateau=True)
    np.testing.assert_allclose(info['tau_ps'],.07,rtol=1e-5)
    np.testing.assert_allclose(info['plateau'],.956,atol=1e-8)


def test_log_decay_exponential_and_cutoff():
    lag=np.arange(0,40.01,.01)
    c=.8+.2*np.exp(-lag/7)
    y,mask,info=fit_log_decay(lag,c,.8,35)
    np.testing.assert_allclose(info['tau_ps'],7,atol=1e-10)
    assert not mask[lag>=35].any()
    assert np.isclose(info['r_squared_centered'],1)
    c[10]=.79
    y,mask,info=fit_log_decay(lag,c,.8,35)
    assert not mask[10] and np.isnan(y[10])


def test_fft_matches_direct_all_lags():
    angles=np.random.default_rng(19).uniform(-180,180,101)
    theta=np.deg2rad(angles)
    direct=np.array([np.cos(theta[:len(theta)-m]-theta[m:]).mean() for m in range(len(theta))])
    np.testing.assert_allclose(cosine_acf(angles),direct,atol=1e-13)


def test_constant_and_wrap_invariance():
    np.testing.assert_allclose(cosine_acf(np.full(25,179.)),1,atol=1e-13)
    angles=np.array([179,-179,178,-178,175.])
    np.testing.assert_allclose(cosine_acf(angles),cosine_acf(angles%360),atol=1e-13)
    assert np.isclose(cosine_acf(angles)[0],1)
