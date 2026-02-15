"""Tests for frequencyfilt module."""

import numpy as np
import pytest

from phasecongruency import (
    filtergrids,
    filtergrid,
    monogenicfilters,
    packedmonogenicfilters,
    lowpassfilter,
    highpassfilter,
    bandpassfilter,
    highboostfilter,
    loggabor,
    gridangles,
    cosineangularfilter,
    gaussianangularfilter,
    perfft2,
    geoseries,
)


class TestFiltergrids:
    def test_shape(self):
        rows, cols = 101, 200
        f, fx, fy = filtergrids(rows, cols)
        assert f.shape == (rows, cols)
        assert fx.shape == (rows, cols)
        assert fy.shape == (rows, cols)

    def test_tuple_input(self):
        rows, cols = 101, 200
        f1, fx1, fy1 = filtergrids(rows, cols)
        f2, fx2, fy2 = filtergrids((rows, cols))
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(fx1, fx2)
        np.testing.assert_array_equal(fy1, fy2)

    def test_dc_at_corner(self):
        f, fx, fy = filtergrids(100, 100)
        assert f[0, 0] == 0.0


class TestFiltergrid:
    def test_shape(self):
        rows, cols = 101, 200
        f = filtergrid(rows, cols)
        assert f.shape == (rows, cols)

    def test_tuple_input(self):
        f1 = filtergrid(101, 200)
        f2 = filtergrid((101, 200))
        np.testing.assert_array_equal(f1, f2)


class TestMonogenicfilters:
    def test_shape(self):
        rows, cols = 101, 200
        H1, H2, f = monogenicfilters(rows, cols)
        assert H1.shape == (rows, cols)
        assert H2.shape == (rows, cols)
        assert f.shape == (rows, cols)

    def test_dc_zero(self):
        H1, H2, f = monogenicfilters(101, 200)
        assert H1[0, 0] == 0
        assert H2[0, 0] == 0
        assert f[0, 0] == 0


class TestPackedmonogenicfilters:
    def test_shape(self):
        rows, cols = 101, 200
        H, f = packedmonogenicfilters(rows, cols)
        assert H.shape == (rows, cols)
        assert f.shape == (rows, cols)


class TestLowpassfilter:
    def test_shape(self):
        sze = (101, 200)
        f = lowpassfilter(sze, 0.2, 2)
        assert f.shape == sze

    def test_invalid_cutoff(self):
        with pytest.raises(ValueError):
            lowpassfilter((100, 100), 0.6, 2)
        with pytest.raises(ValueError):
            lowpassfilter((100, 100), -0.1, 2)


class TestBandpassfilter:
    def test_shape(self):
        sze = (101, 200)
        f = bandpassfilter(sze, 0.1, 0.2, 2)
        assert f.shape == sze


class TestHighboostfilter:
    def test_shape(self):
        sze = (101, 200)
        f = highboostfilter(sze, 0.2, 2, 2)
        assert f.shape == sze


class TestHighpassfilter:
    def test_shape(self):
        sze = (101, 200)
        f = highpassfilter(sze, 0.2, 2)
        assert f.shape == sze


class TestLoggabor:
    def test_zero_frequency(self):
        fo = 0.3
        sigmaOnf = 0.55
        assert loggabor(0, fo, sigmaOnf) == 0.0

    def test_center_frequency(self):
        fo = 0.3
        sigmaOnf = 0.55
        assert abs(loggabor(fo, fo, sigmaOnf) - 1.0) < np.finfo(float).eps


class TestGridangles:
    def test_shape(self):
        rows, cols = 101, 200
        f, fx, fy = filtergrids(rows, cols)
        sintheta, costheta = gridangles(f, fx, fy)
        assert sintheta.shape == (rows, cols)
        assert costheta.shape == (rows, cols)


class TestCosineangularfilter:
    def test_shape(self):
        rows, cols = 101, 200
        f, fx, fy = filtergrids(rows, cols)
        sintheta, costheta = gridangles(f, fx, fy)
        angl = np.pi / 4
        wavelen = np.pi / 8
        fltr = cosineangularfilter(angl, wavelen, sintheta, costheta)
        assert fltr.shape == (rows, cols)


class TestGaussianangularfilter:
    def test_shape(self):
        rows, cols = 101, 200
        f, fx, fy = filtergrids(rows, cols)
        sintheta, costheta = gridangles(f, fx, fy)
        angl = np.pi / 4
        thetaSigma = 0.4
        fltr = gaussianangularfilter(angl, thetaSigma, sintheta, costheta)
        assert fltr.shape == (rows, cols)


class TestPerfft2:
    def test_decomposition(self):
        """Test that p + s = original image."""
        np.random.seed(42)
        img = np.random.randn(64, 64)
        P, S, p, s = perfft2(img)
        np.testing.assert_allclose(p + s, img, atol=1e-10)

    def test_shape(self):
        img = np.random.randn(64, 64)
        P, S, p, s = perfft2(img)
        assert P.shape == img.shape
        assert S.shape == img.shape
        assert p.shape == img.shape
        assert s.shape == img.shape


class TestGeoseries:
    def test_basic(self):
        s = geoseries(0.5, 2, 4)
        expected = np.array([0.5, 1.0, 2.0, 4.0])
        np.testing.assert_allclose(s, expected, atol=np.finfo(float).eps)

    def test_tuple_input(self):
        s = geoseries((0.5, 4), 4)
        expected = np.array([0.5, 1.0, 2.0, 4.0])
        np.testing.assert_allclose(s, expected, atol=np.finfo(float).eps)

    def test_equivalence(self):
        s1 = geoseries(0.5, 2, 4)
        s2 = geoseries((0.5, 4), 4)
        np.testing.assert_allclose(s1, s2, atol=np.finfo(float).eps)

    def test_tuple_single_element(self):
        s = geoseries((0.5, 4), 1)
        np.testing.assert_allclose(s, np.array([0.5]), atol=np.finfo(float).eps)
