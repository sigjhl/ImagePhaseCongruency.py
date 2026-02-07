"""Tests for utilities module."""

import numpy as np
import pytest

from phasecongruency import replacenan, hysthresh, imgnormalize, imgnormalise, histtruncate


class TestReplacenan:
    def test_replacenan_default(self):
        img = np.ones((30, 30))
        img[5:20, 10:15] = np.nan
        newimg, mask = replacenan(img)
        assert not np.any(np.isnan(newimg))
        assert np.all(newimg[5:20, 10:15] == 0)
        assert np.all(mask[5:20, 10:15] == False)
        assert np.all(mask[0:5, :] == True)

    def test_replacenan_custom_value(self):
        img = np.ones((30, 30))
        img[5:20, 10:15] = np.nan
        newimg, mask = replacenan(img, 2)
        assert np.all(newimg[5:20, 10:15] == 2)


class TestHystthresh:
    def test_basic_thresholding(self):
        img = np.zeros((30, 30))
        img[3:25, 15] = 5
        img[8:12, 15] = 10
        img[15:20, 15] = 10
        img[15, 15] = 20

        bw = hysthresh(img, 8, 20)

        expected = np.zeros((30, 30), dtype=bool)
        expected[15:20, 15] = True

        np.testing.assert_array_equal(bw, expected)

    def test_swapped_thresholds(self):
        """T1 and T2 can be in any order."""
        img = np.zeros((30, 30))
        img[15:20, 15] = 10
        img[15, 15] = 20

        bw1 = hysthresh(img, 8, 20)
        bw2 = hysthresh(img, 20, 8)
        np.testing.assert_array_equal(bw1, bw2)


class TestImgnormalize:
    def test_normalize_01(self):
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = imgnormalize(img)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)

    def test_normalize_mean_var(self):
        img = np.random.randn(100, 100)
        reqmean = 5.0
        reqvar = 2.0
        result = imgnormalise(img, reqmean, reqvar)
        assert np.isclose(np.mean(result), reqmean, atol=0.01)
        assert np.isclose(np.var(result), reqvar, atol=0.1)


class TestHisttruncate:
    def test_basic(self):
        img = np.arange(100, dtype=float).reshape(10, 10)
        result = histtruncate(img, 10, 10)
        assert result.min() >= img.ravel()[10]
        assert result.max() <= img.ravel()[89]

    def test_symmetric(self):
        img = np.arange(100, dtype=float).reshape(10, 10)
        result = histtruncate(img, 5)
        assert result.min() >= 0
        assert result.max() <= 99

    def test_invalid_values(self):
        img = np.ones((10, 10))
        with pytest.raises(ValueError):
            histtruncate(img, -1, 10)
        with pytest.raises(ValueError):
            histtruncate(img, 10, 101)

    def test_3d_raises(self):
        img = np.ones((10, 10, 3))
        with pytest.raises(ValueError):
            histtruncate(img, 5, 5)
