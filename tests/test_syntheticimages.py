"""Tests for syntheticimages module."""

import numpy as np
import pytest

from phasecongruency import (
    step2line,
    circsine,
    starsine,
    noiseonf,
    nophase,
    quantizephase,
    swapphase,
)


class TestStep2line:
    def test_default(self):
        img = step2line()
        assert img.shape == (512, 512)
        assert img.dtype == np.float64

    def test_custom_size(self):
        img = step2line(128)
        assert img.shape == (128, 128)

    def test_custom_params(self):
        img = step2line(128, nscales=10, ampexponent=-1.5,
                        ncycles=3, phasecycles=3)
        assert img.shape == (128, 128)


class TestCircsine:
    def test_default(self):
        img = circsine()
        assert img.shape == (512, 512)

    def test_custom(self):
        img = circsine(128, wavelength=20, nscales=10)
        assert img.shape == (128, 128)

    def test_odd_p_raises(self):
        with pytest.raises(ValueError):
            circsine(128, p=3)

    def test_trim_masks_corners(self):
        img_untrim = circsine(64, wavelength=7, nscales=5, trim=False)
        img_trim = circsine(64, wavelength=7, nscales=5, trim=True)
        assert not np.allclose(img_untrim, img_trim)
        assert np.isclose(img_trim[0, 0], 0.0)


class TestStarsine:
    def test_default(self):
        img = starsine()
        assert img.shape == (512, 512)

    def test_custom(self):
        img = starsine(128, ncycles=5, nscales=10)
        assert img.shape == (128, 128)


class TestNoiseonf:
    def test_scalar(self):
        np.random.seed(42)
        img = noiseonf(128, 1.5)
        assert img.shape == (128, 128)

    def test_tuple(self):
        np.random.seed(42)
        img = noiseonf((64, 128), 1.0)
        assert img.shape == (64, 128)


class TestNophase:
    def test_shape_preserved(self):
        np.random.seed(42)
        img = np.random.randn(64, 64)
        result = nophase(img)
        assert result.shape == img.shape


class TestQuantizephase:
    def test_shape_preserved(self):
        np.random.seed(42)
        img = np.random.randn(64, 64)
        result = quantizephase(img, 4)
        assert result.shape == img.shape


class TestSwapphase:
    def test_basic(self):
        np.random.seed(42)
        img1 = np.random.randn(64, 64)
        img2 = np.random.randn(64, 64)
        newimg1, newimg2 = swapphase(img1, img2)
        assert newimg1.shape == img1.shape
        assert newimg2.shape == img2.shape
        # Normalized to 0-1
        assert newimg1.min() >= -1e-10
        assert newimg1.max() <= 1 + 1e-10

    def test_size_mismatch(self):
        with pytest.raises(ValueError):
            swapphase(np.ones((10, 10)), np.ones((10, 20)))
