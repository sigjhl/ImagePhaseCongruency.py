"""Tests for phasecongruency module.

Hard to test these functions other than visually. This script simply
runs them all to make sure that they at least run without error.
"""

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
