"""Tests for phasecongruency module.

Hard to test these functions other than visually. This script simply
runs them all to make sure that they at least run without error.
"""

import importlib.util
import json

import numpy as np
import pytest

from phasecongruency import (
    ppdrc,
    phasecongmono,
    phasesymmono,
    phasecong3,
    phasesym,
    ppdenoise,
    monofilt,
    gaborconvolve,
    highpassmonogenic,
    bandpassmonogenic,
    geoseries,
)
import phasecongruency.phasecongruency as phasecongruency_module


@pytest.fixture
def test_image():
    """Create a simple test image."""
    np.random.seed(42)
    return np.random.randn(64, 64)


class TestPpdrc:
    def test_single_wavelength(self, test_image):
        dimg = ppdrc(test_image, 50, clip=0.01, n=2)
        assert dimg.shape == test_image.shape

    def test_multiple_wavelengths(self, test_image):
        dimg = ppdrc(test_image, geoseries((20, 60), 3))
        assert isinstance(dimg, list)
        assert len(dimg) == 3

    def test_multiple_wavelengths_sets_fortran_linear_anchor_pixels(self, test_image):
        dimg = ppdrc(test_image, geoseries((20, 60), 3))
        maxrange = max(np.max(np.abs(d)) for d in dimg)
        for d in dimg:
            assert np.isclose(d[0, 0], maxrange)
            assert np.isclose(d[1, 0], -maxrange)


class TestPhasecongmono:
    def test_default(self, test_image):
        PC, or_, ft, T = phasecongmono(test_image)
        assert PC.shape == test_image.shape
        assert or_.shape == test_image.shape
        assert ft.shape == test_image.shape

    def test_custom_params(self, test_image):
        PC, or_, ft, T = phasecongmono(
            test_image, nscale=3, minwavelength=3, mult=2,
            sigmaonf=0.55, k=3, cutoff=0.5, g=10,
            deviationgain=1.5, noisemethod=-1
        )
        assert PC.shape == test_image.shape

    def test_nscale_too_small_raises(self, test_image):
        with pytest.raises(ValueError):
            phasecongmono(test_image, nscale=1)

    def test_invalid_backend_raises(self, test_image):
        with pytest.raises(ValueError):
            phasecongmono(test_image, backend="invalid-backend")

    def test_torch_backend_requires_torch_if_unavailable(self, test_image):
        if importlib.util.find_spec("torch") is not None:
            pytest.skip("torch is installed")
        with pytest.raises(ImportError):
            phasecongmono(test_image, backend="torch")

    def test_auto_backend_falls_back_to_numpy_if_torch_unavailable(self, test_image):
        if importlib.util.find_spec("torch") is not None:
            pytest.skip("torch is installed")
        PC, or_, ft, _ = phasecongmono(test_image, backend="auto")
        assert PC.shape == test_image.shape
        assert or_.shape == test_image.shape
        assert ft.shape == test_image.shape

    def test_auto_backend_persists_and_reuses_cache_when_torch_available(
        self, test_image, tmp_path, monkeypatch
    ):
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch is not installed")

        cache_path = tmp_path / "auto_backend_cache.json"
        monkeypatch.setenv("PHASECONGRUENCY_AUTO_CACHE_PATH", str(cache_path))
        monkeypatch.setattr(phasecongruency_module, "_AUTO_CACHE_PATH", None)
        monkeypatch.setattr(phasecongruency_module, "_AUTO_BACKEND_CACHE", None)

        img = test_image[:32, :32]
        PC, or_, ft, _ = phasecongmono(img, backend="auto")
        assert PC.shape == img.shape
        assert or_.shape == img.shape
        assert ft.shape == img.shape
        assert cache_path.exists()

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert payload.get("entries")

        monkeypatch.setattr(phasecongruency_module, "_AUTO_BACKEND_CACHE", None)

        def fail_benchmark(*args, **kwargs):
            raise AssertionError("benchmark should not run on cache hit")

        monkeypatch.setattr(
            phasecongruency_module, "_benchmark_callable", fail_benchmark
        )
        phasecongmono(img, backend="auto")


class TestPhasesymmono:
    def test_default(self, test_image):
        phSym, symEnergy, T = phasesymmono(test_image)
        assert phSym.shape == test_image.shape
        assert symEnergy.shape == test_image.shape

    def test_custom_params(self, test_image):
        phSym, symEnergy, T = phasesymmono(
            test_image, nscale=3, minwavelength=3, mult=2,
            sigmaonf=0.55, k=2, polarity=1, noisemethod=-1
        )
        assert phSym.shape == test_image.shape


class TestPhasecong3:
    def test_default(self, test_image):
        M, m, or_, ft, EO, T = phasecong3(test_image)
        assert M.shape == test_image.shape
        assert m.shape == test_image.shape
        assert or_.shape == test_image.shape
        assert ft.shape == test_image.shape

    def test_custom_params(self, test_image):
        M, m, or_, ft, EO, T = phasecong3(
            test_image, nscale=3, norient=6, minwavelength=3,
            mult=2, sigmaonf=0.55, k=2, cutoff=0.5,
            g=10, noisemethod=-1
        )
        assert M.shape == test_image.shape

    def test_nscale_too_small_raises(self, test_image):
        with pytest.raises(ValueError):
            phasecong3(test_image, nscale=1)

    def test_return_eo_false_matches_core_outputs_numpy(self, test_image):
        M1, m1, or1, ft1, EO1, T1 = phasecong3(test_image, backend="numpy", return_eo=True)
        M2, m2, or2, ft2, EO2, T2 = phasecong3(test_image, backend="numpy", return_eo=False)

        assert EO1 is not None
        assert EO2 is None
        np.testing.assert_allclose(M1, M2, atol=1e-12)
        np.testing.assert_allclose(m1, m2, atol=1e-12)
        np.testing.assert_allclose(or1, or2, atol=1e-12)
        np.testing.assert_allclose(ft1, ft2, atol=1e-12)
        assert np.isclose(T1, T2)

    def test_return_eo_false_matches_core_outputs_torch(self, test_image):
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch is not installed")

        M1, m1, or1, ft1, EO1, T1 = phasecong3(test_image, backend="torch", return_eo=True)
        M2, m2, or2, ft2, EO2, T2 = phasecong3(test_image, backend="torch", return_eo=False)

        assert EO1 is not None
        assert EO2 is None
        np.testing.assert_allclose(M1, M2, atol=1e-7)
        np.testing.assert_allclose(m1, m2, atol=1e-7)
        np.testing.assert_allclose(or1, or2, atol=1e-7)
        np.testing.assert_allclose(ft1, ft2, atol=1e-7)
        assert np.isclose(T1, T2)

    def test_return_eo_false_matches_core_outputs_torch_mps(self, test_image):
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch is not installed")
        import torch

        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            pytest.skip("torch MPS is not available")

        M1, m1, or1, ft1, EO1, T1 = phasecong3(test_image, backend="torch-mps", return_eo=True)
        M2, m2, or2, ft2, EO2, T2 = phasecong3(test_image, backend="torch-mps", return_eo=False)

        assert EO1 is not None
        assert EO2 is None
        np.testing.assert_allclose(M1, M2, atol=1e-5)
        np.testing.assert_allclose(m1, m2, atol=1e-5)
        np.testing.assert_allclose(or1, or2, atol=1e-5)
        np.testing.assert_allclose(ft1, ft2, atol=1e-5)
        assert np.isclose(T1, T2)


class TestPhasesym:
    def test_default(self, test_image):
        phSym, orient, totalEnergy, T = phasesym(test_image)
        assert phSym.shape == test_image.shape
        assert orient.shape == test_image.shape
        assert totalEnergy.shape == test_image.shape

    def test_custom_params(self, test_image):
        phSym, orient, totalEnergy, T = phasesym(
            test_image, nscale=3, norient=4, minwavelength=3,
            mult=2, sigmaonf=0.55, k=2, polarity=0, noisemethod=-1
        )
        assert phSym.shape == test_image.shape


class TestPpdenoise:
    def test_default(self, test_image):
        clean = ppdenoise(test_image)
        assert clean.shape == test_image.shape

    def test_custom_params(self, test_image):
        clean = ppdenoise(
            test_image, nscale=3, norient=4,
            mult=2.5, minwavelength=2, sigmaonf=0.55,
            dthetaonsigma=1.0, k=3, softness=1.0
        )
        assert clean.shape == test_image.shape


class TestMonofilt:
    def test_basic(self, test_image):
        f, h1f, h2f, A, theta, psi = monofilt(
            test_image, nscale=3, minWaveLength=3,
            mult=2, sigmaOnf=0.55
        )
        assert len(f) == 3
        assert f[0].shape == test_image.shape
        assert A[0].shape == test_image.shape


class TestGaborconvolve:
    def test_basic(self, test_image):
        EO, BP = gaborconvolve(
            test_image, nscale=3, norient=4,
            minWaveLength=3, mult=2,
            sigmaOnf=0.55, dThetaOnSigma=1.3
        )
        assert len(EO) == 3
        assert len(EO[0]) == 4
        assert len(BP) == 3
        assert BP[0].shape == test_image.shape


class TestHighpassmonogenic:
    def test_single_wavelength(self, test_image):
        ph, orient, E = highpassmonogenic(test_image, 20, 4)
        assert ph.shape == test_image.shape
        assert orient.shape == test_image.shape
        assert E.shape == test_image.shape


class TestBandpassmonogenic:
    def test_single_wavelength(self, test_image):
        ph, orient, E = bandpassmonogenic(test_image, 4, 20, 4)
        assert ph.shape == test_image.shape
        assert orient.shape == test_image.shape
        assert E.shape == test_image.shape
