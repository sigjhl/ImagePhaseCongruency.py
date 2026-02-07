"""
Cross-validation test: verify the Python transpilation matches the Julia
implementation by checking mathematical invariants and reimplementing
critical computations from scratch.

This validates correctness by:
1. Reimplementing key Julia algorithms independently and comparing
2. Checking mathematical properties that both implementations must satisfy
3. Using known analytical results as ground truth
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
import sys

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def ref_filtergrids(rows, cols):
    """Reference implementation of filtergrids, translated line-by-line from Julia."""
    if cols % 2 == 1:
        fxrange = np.arange(-(cols-1)/2, (cols-1)/2 + 1) / cols
    else:
        fxrange = np.arange(-cols/2, cols/2) / cols

    if rows % 2 == 1:
        fyrange = np.arange(-(rows-1)/2, (rows-1)/2 + 1) / rows
    else:
        fyrange = np.arange(-rows/2, rows/2) / rows

    # Julia: fx = [c for r = fyrange, c = fxrange]  (column-major)
    # In Julia, `[c for r = fyrange, c = fxrange]` creates a matrix where
    # the first index iterates over fyrange and second over fxrange.
    # This is equivalent to meshgrid in Python.
    fx = np.zeros((rows, cols))
    fy = np.zeros((rows, cols))
    for i, r in enumerate(fyrange):
        for j, c in enumerate(fxrange):
            fx[i, j] = c
            fy[i, j] = r

    fx = ifftshift(fx)
    fy = ifftshift(fy)
    f = np.sqrt(fx**2 + fy**2)
    return f, fx, fy


def ref_filtergrid(rows, cols):
    """Reference implementation of filtergrid from Julia."""
    if cols % 2 == 1:
        fxrange = np.arange(-(cols-1)/2, (cols-1)/2 + 1) / cols
    else:
        fxrange = np.arange(-cols/2, cols/2) / cols

    if rows % 2 == 1:
        fyrange = np.arange(-(rows-1)/2, (rows-1)/2 + 1) / rows
    else:
        fyrange = np.arange(-rows/2, rows/2) / rows

    f = np.zeros((rows, cols))
    for i, fy in enumerate(fyrange):
        for j, fx in enumerate(fxrange):
            f[i, j] = np.sqrt(fx**2 + fy**2)

    return ifftshift(f)


def ref_loggabor(f, fo, sigmaOnf):
    """Reference implementation of loggabor from Julia."""
    if f < np.finfo(float).eps:
        return 0.0
    else:
        return np.exp((-(np.log(f/fo))**2) / (2 * np.log(sigmaOnf)**2))


def ref_lowpassfilter_scalar(f, cutoff, n):
    """Reference scalar lowpassfilter from Julia."""
    return 1.0 / (1.0 + (f / cutoff)**(2*n))


def ref_step2line(sze, nscales=50, ampexponent=-1, ncycles=1.5, phasecycles=0.25):
    """Reference implementation of step2line from Julia.

    Julia code (0-indexed equivalent):
        x = (0:(sze-1))/(sze-1)*ncycles*2*pi
        for row = 1:sze
            for scale = 1:2:(nscales*2-1)
                img[row,:] += scale^ampexponent * sin(scale*x + phaseoffset)
            phaseoffset += phasecycles*2*pi/sze
    """
    x = np.arange(sze) / (sze - 1) * ncycles * 2 * np.pi
    img = np.zeros((sze, sze))
    phaseoffset = 0.0

    for row in range(sze):
        for scale in range(1, nscales * 2, 2):
            img[row, :] += scale ** float(ampexponent) * np.sin(scale * x + phaseoffset)
        phaseoffset += phasecycles * 2 * np.pi / sze

    return img


def ref_perfft2(img):
    """Reference implementation of perfft2 from Julia."""
    rows, cols = img.shape

    s = np.zeros((rows, cols))
    s[0, :] = img[0, :] - img[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + img[:, 0] - img[:, -1]
    s[:, -1] = s[:, -1] - img[:, 0] + img[:, -1]

    cxrange = 2 * np.pi * np.arange(cols) / cols
    cyrange = 2 * np.pi * np.arange(rows) / rows

    denom = np.zeros((rows, cols))
    for i, cy in enumerate(cyrange):
        for j, cx in enumerate(cxrange):
            denom[i, j] = 2 * (2 - np.cos(cx) - np.cos(cy))

    S = fft2(s) / denom
    S[0, 0] = 0.0

    P = fft2(img) - S

    s_out = np.real(ifft2(S))
    p_out = img - s_out

    return P, S, p_out, s_out


def ref_monogenicfilters(rows, cols):
    """Reference implementation of monogenicfilters from Julia."""
    f, fx, fy = ref_filtergrids(rows, cols)
    f[0, 0] = 1  # avoid divide by zero

    H1 = 1j * fx / f
    H2 = 1j * fy / f

    H1[0, 0] = 0
    H2[0, 0] = 0
    f[0, 0] = 0
    return H1, H2, f


def ref_highpassmonogenic(img, maxwavelength, n):
    """Reference implementation of highpassmonogenic from Julia."""
    IMG = fft2(img)
    H1, H2, freq = ref_monogenicfilters(*img.shape)

    H = 1.0 - 1.0 / (1.0 + (freq * maxwavelength)**(2*n))
    f = np.real(ifft2(H * IMG))
    h1f = np.real(ifft2(H * H1 * IMG))
    h2f = np.real(ifft2(H * H2 * IMG))

    phase = np.arctan2(f, np.sqrt(h1f**2 + h2f**2 + np.finfo(float).eps))
    orient = np.arctan2(h2f, h1f)
    E = np.sqrt(f**2 + h1f**2 + h2f**2)

    return phase, orient, E


def ref_phasecongmono(img, nscale=4, minwavelength=3, mult=2.1, sigmaonf=0.55,
                      k=3.0, noisemethod=-1, cutoff=0.5, g=10.0, deviationgain=1.5):
    """Reference implementation of phasecongmono from Julia - loop-by-loop."""
    epsilon = 0.0001
    rows, cols = img.shape
    IMG = fft2(img)

    sumAn = np.zeros((rows, cols))
    sumf = np.zeros((rows, cols))
    sumh1 = np.zeros((rows, cols))
    sumh2 = np.zeros((rows, cols))
    maxAn = np.zeros((rows, cols))

    tau = 0.0
    T = 0.0

    # Packed monogenic filters - reference implementation
    f_grid, fx, fy = ref_filtergrids(rows, cols)
    f_grid[0, 0] = 1
    H = (1j * fx - fy) / f_grid
    H[0, 0] = 0
    f_grid[0, 0] = 0
    freq = f_grid

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        IMGF = np.zeros((rows, cols), dtype=complex)
        for i in range(rows):
            for j in range(cols):
                IMGF[i, j] = (
                    IMG[i, j]
                    * ref_loggabor(freq[i, j], fo, sigmaonf)
                    * ref_lowpassfilter_scalar(freq[i, j], 0.45, 15)
                )

        f = np.real(ifft2(IMGF))
        h = ifft2(IMGF * H)
        h1 = np.real(h)
        h2 = np.imag(h)
        An = np.sqrt(f**2 + h1**2 + h2**2)
        sumAn += An
        sumf += f
        sumh1 += h1
        sumh2 += h2

        if s == 0:
            if abs(noisemethod + 1) < epsilon:
                tau = np.median(sumAn) / np.sqrt(np.log(4))
            elif abs(noisemethod + 2) < epsilon:
                pass  # rayleighmode
            maxAn = An.copy()
        else:
            maxAn = np.maximum(maxAn, An)

    width = (sumAn / (maxAn + epsilon) - 1) / (nscale - 1)
    weight = 1.0 / (1 + np.exp((cutoff - width) * g))

    if noisemethod >= 0:
        T = noisemethod
    else:
        totalTau = tau * (1 - (1/mult)**nscale) / (1 - (1/mult))
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        T = EstNoiseEnergyMean + k * EstNoiseEnergySigma

    or_ = np.arctan2(-sumh2, sumh1)
    ft = np.arctan2(sumf, np.sqrt(sumh1**2 + sumh2**2))
    energy = np.sqrt(sumf**2 + sumh1**2 + sumh2**2)

    PC = (
        weight
        * np.maximum(1 - deviationgain * np.arccos(energy / (sumAn + epsilon)), 0)
        * np.maximum(energy - T, 0)
        / (energy + epsilon)
    )

    return PC, or_, ft, T


# ============================================================
# Import the package under test
# ============================================================
from phasecongruency import (
    filtergrids, filtergrid, monogenicfilters, packedmonogenicfilters,
    loggabor, lowpassfilter, highpassfilter, bandpassfilter, highboostfilter,
    gridangles, cosineangularfilter, gaussianangularfilter,
    perfft2, geoseries,
    step2line, circsine, starsine, noiseonf, nophase, quantizephase, swapphase,
    phasecongmono, phasesymmono, ppdrc, phasecong3, phasesym, ppdenoise,
    monofilt, gaborconvolve, highpassmonogenic, bandpassmonogenic,
    replacenan, fillnan, hysthresh, imgnormalize, histtruncate,
)


def test_filtergrids():
    print("\n=== filtergrids ===")
    for rows, cols in [(101, 200), (64, 64), (63, 63), (128, 100)]:
        f, fx, fy = filtergrids(rows, cols)
        rf, rfx, rfy = ref_filtergrids(rows, cols)
        check(f"filtergrids({rows},{cols}) f matches ref",
              np.allclose(f, rf, atol=1e-14))
        check(f"filtergrids({rows},{cols}) fx matches ref",
              np.allclose(fx, rfx, atol=1e-14))
        check(f"filtergrids({rows},{cols}) fy matches ref",
              np.allclose(fy, rfy, atol=1e-14))
        check(f"filtergrids({rows},{cols}) DC at [0,0] is 0",
              f[0, 0] == 0.0)
        # Max frequency at corners is sqrt(0.5^2 + 0.5^2) ~ 0.707
        check(f"filtergrids({rows},{cols}) f max <= sqrt(0.5)",
              np.max(f) <= np.sqrt(0.5) + 1e-10)


def test_filtergrid():
    print("\n=== filtergrid ===")
    for rows, cols in [(101, 200), (64, 64), (63, 63)]:
        f = filtergrid(rows, cols)
        rf = ref_filtergrid(rows, cols)
        check(f"filtergrid({rows},{cols}) matches ref",
              np.allclose(f, rf, atol=1e-14))


def test_monogenicfilters():
    print("\n=== monogenicfilters ===")
    for rows, cols in [(64, 64), (101, 200)]:
        H1, H2, f = monogenicfilters(rows, cols)
        rH1, rH2, rf = ref_monogenicfilters(rows, cols)
        check(f"monogenicfilters({rows},{cols}) H1 matches ref",
              np.allclose(H1, rH1, atol=1e-14))
        check(f"monogenicfilters({rows},{cols}) H2 matches ref",
              np.allclose(H2, rH2, atol=1e-14))
        check(f"monogenicfilters({rows},{cols}) f matches ref",
              np.allclose(f, rf, atol=1e-14))
        check(f"monogenicfilters({rows},{cols}) DC=0",
              H1[0,0] == 0 and H2[0,0] == 0 and f[0,0] == 0)


def test_loggabor():
    print("\n=== loggabor ===")
    fo = 0.3
    sigma = 0.55
    check("loggabor(0) == 0", loggabor(0, fo, sigma) == 0.0)
    check("loggabor(fo) == 1", abs(loggabor(fo, fo, sigma) - 1.0) < 1e-15)
    check("loggabor matches ref at 0.1",
          abs(loggabor(0.1, fo, sigma) - ref_loggabor(0.1, fo, sigma)) < 1e-15)
    check("loggabor matches ref at 0.5",
          abs(loggabor(0.5, fo, sigma) - ref_loggabor(0.5, fo, sigma)) < 1e-15)
    # Symmetry on log scale
    v1 = loggabor(fo * 2, fo, sigma)
    v2 = loggabor(fo / 2, fo, sigma)
    check("loggabor log-symmetric", abs(v1 - v2) < 1e-15)


def test_lowpassfilter():
    print("\n=== lowpassfilter ===")
    # Scalar version
    check("lowpass(0) == 1", abs(lowpassfilter(0.0, 0.25, 2) - 1.0) < 1e-15)
    check("lowpass(cutoff) == 0.5", abs(lowpassfilter(0.25, 0.25, 1) - 0.5) < 1e-10)
    # High frequency -> 0
    check("lowpass(high_f) ~ 0", lowpassfilter(10.0, 0.25, 10) < 1e-30)
    # Array version
    f = lowpassfilter((64, 64), 0.25, 2)
    check("lowpass array DC = 1.0", abs(f[0, 0] - 1.0) < 1e-10)
    check("lowpass array values in [0,1]", np.all(f >= 0) and np.all(f <= 1))
    # high + low = 1
    lp = lowpassfilter((64, 64), 0.25, 2)
    hp = highpassfilter((64, 64), 0.25, 2)
    check("lowpass + highpass = 1", np.allclose(lp + hp, 1.0, atol=1e-14))


def test_bandpassfilter():
    print("\n=== bandpassfilter ===")
    bp = bandpassfilter((64, 64), 0.1, 0.3, 2)
    check("bandpass DC = 0", abs(bp[0, 0]) < 1e-10)
    check("bandpass values in [0,1]", np.all(bp >= -1e-10) and np.all(bp <= 1 + 1e-10))


def test_perfft2():
    print("\n=== perfft2 ===")
    np.random.seed(42)
    img = np.random.randn(64, 64)
    P, S, p, s = perfft2(img)
    rP, rS, rp, rs = ref_perfft2(img)

    check("perfft2 p+s = img", np.allclose(p + s, img, atol=1e-10))
    check("perfft2 p matches ref", np.allclose(p, rp, atol=1e-10))
    check("perfft2 s matches ref", np.allclose(s, rs, atol=1e-10))
    check("perfft2 P matches ref", np.allclose(P, rP, atol=1e-6))
    check("perfft2 S mean = 0", abs(S[0, 0]) < 1e-10)


def test_geoseries():
    print("\n=== geoseries ===")
    s = geoseries(0.5, 2, 4)
    check("geoseries basic", np.allclose(s, [0.5, 1.0, 2.0, 4.0], atol=1e-14))
    s2 = geoseries((0.5, 4.0), 4)
    check("geoseries tuple", np.allclose(s, s2, atol=1e-14))
    s3 = geoseries(1.0, 3, 5)
    check("geoseries mult=3", np.allclose(s3, [1, 3, 9, 27, 81], atol=1e-10))


def test_step2line():
    print("\n=== step2line ===")
    img = step2line(64, nscales=10)
    ref = ref_step2line(64, nscales=10)
    check("step2line matches ref (64, nscales=10)",
          np.allclose(img, ref, atol=1e-12))

    img2 = step2line(128, nscales=20, ampexponent=-1.5, ncycles=3, phasecycles=0.5)
    ref2 = ref_step2line(128, nscales=20, ampexponent=-1.5, ncycles=3, phasecycles=0.5)
    check("step2line matches ref (128, custom params)",
          np.allclose(img2, ref2, atol=1e-12))


def test_highpassmonogenic():
    print("\n=== highpassmonogenic ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)

    ph, orient, E = highpassmonogenic(img, 20, 4)
    rph, rorient, rE = ref_highpassmonogenic(img, 20, 4)

    check("highpassmonogenic phase matches ref",
          np.allclose(ph, rph, atol=1e-10))
    check("highpassmonogenic orient matches ref",
          np.allclose(orient, rorient, atol=1e-10))
    check("highpassmonogenic E matches ref",
          np.allclose(E, rE, atol=1e-10))
    check("highpassmonogenic E >= 0", np.all(E >= -1e-10))


def test_phasecongmono():
    print("\n=== phasecongmono ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)

    PC, or_, ft, T = phasecongmono(img, nscale=3, minwavelength=3, mult=2,
                                    sigmaonf=0.55, k=3, noisemethod=-1)
    rPC, ror, rft, rT = ref_phasecongmono(img, nscale=3, minwavelength=3, mult=2,
                                            sigmaonf=0.55, k=3, noisemethod=-1)

    check("phasecongmono PC matches ref",
          np.allclose(PC, rPC, atol=1e-10),
          f"max diff={np.max(np.abs(PC - rPC)):.2e}")
    check("phasecongmono or matches ref",
          np.allclose(or_, ror, atol=1e-10),
          f"max diff={np.max(np.abs(or_ - ror)):.2e}")
    check("phasecongmono ft matches ref",
          np.allclose(ft, rft, atol=1e-10),
          f"max diff={np.max(np.abs(ft - rft)):.2e}")
    check("phasecongmono T matches ref",
          abs(T - rT) < 1e-10,
          f"T={T}, rT={rT}")
    check("phasecongmono PC in [0,1]",
          np.all(PC >= -1e-10) and np.all(PC <= 1 + 1e-10))


def test_ppdrc():
    print("\n=== ppdrc ===")
    np.random.seed(42)
    img = np.random.randn(32, 32) * 100  # Scale up as recommended
    dimg = ppdrc(img, 20, clip=0.01, n=2)
    check("ppdrc returns 2D array", dimg.ndim == 2)
    check("ppdrc shape matches input", dimg.shape == img.shape)

    # Multiple wavelengths
    dimgs = ppdrc(img, geoseries((10, 30), 3))
    check("ppdrc multi returns list", isinstance(dimgs, list))
    check("ppdrc multi length", len(dimgs) == 3)


def test_phasesymmono():
    print("\n=== phasesymmono ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    phSym, symE, T = phasesymmono(img, nscale=3)
    check("phasesymmono phSym shape", phSym.shape == img.shape)
    check("phasesymmono phSym >= 0", np.all(phSym >= -1e-10))
    check("phasesymmono T > 0", T > 0)


def test_phasecong3():
    print("\n=== phasecong3 ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    M, m, or_, ft, EO, T = phasecong3(img, nscale=3, norient=4)
    check("phasecong3 M shape", M.shape == img.shape)
    check("phasecong3 m shape", m.shape == img.shape)
    check("phasecong3 M >= m (pointwise)", np.all(M >= m - 1e-10))
    check("phasecong3 M >= 0", np.all(M >= -1e-10))
    check("phasecong3 EO structure", len(EO) == 3 and len(EO[0]) == 4)


def test_phasesym():
    print("\n=== phasesym ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    phSym, orient, totE, T = phasesym(img, nscale=3, norient=4)
    check("phasesym phSym shape", phSym.shape == img.shape)
    check("phasesym phSym >= 0", np.all(phSym >= -1e-10))


def test_ppdenoise():
    print("\n=== ppdenoise ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    clean = ppdenoise(img, nscale=3, norient=4)
    check("ppdenoise shape", clean.shape == img.shape)
    check("ppdenoise is real", np.all(np.isreal(clean)))


def test_monofilt():
    print("\n=== monofilt ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    f, h1f, h2f, A, theta, psi = monofilt(img, 3, 3, 2, 0.55)
    check("monofilt nscale arrays", len(f) == 3)
    check("monofilt A >= 0", all(np.all(a >= -1e-10) for a in A))
    # Energy = sqrt(f^2 + h1f^2 + h2f^2) should equal A
    for s in range(3):
        E = np.sqrt(f[s]**2 + h1f[s]**2 + h2f[s]**2)
        check(f"monofilt A[{s}] = sqrt(f^2+h1^2+h2^2)",
              np.allclose(A[s], E, atol=1e-10))


def test_gaborconvolve():
    print("\n=== gaborconvolve ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    EO, BP = gaborconvolve(img, 3, 4, 3, 2, 0.55, 1.3)
    check("gaborconvolve EO shape", len(EO) == 3 and len(EO[0]) == 4)
    check("gaborconvolve BP shape", len(BP) == 3)
    check("gaborconvolve BP are real", all(np.all(np.isreal(b)) for b in BP))


def test_syntheticimages():
    print("\n=== syntheticimages ===")
    # circsine: at center, radius=0, all sin terms evaluate to sin(offset)
    img = circsine(65, wavelength=40, nscales=1, offset=np.pi/2)
    center = img[32, 32]
    expected = np.sin(np.pi/2)  # = 1.0  (only 1 scale, scale=1)
    check("circsine center value with offset pi/2",
          abs(center - expected) < 1e-10,
          f"got {center}, expected {expected}")

    # starsine: angular pattern
    img = starsine(64, ncycles=10, nscales=1, offset=0)
    check("starsine shape", img.shape == (64, 64))

    # noiseonf: with p=0, should be close to raw Gaussian noise
    np.random.seed(42)
    img = noiseonf(64, 0)
    check("noiseonf p=0 shape", img.shape == (64, 64))


def test_utilities():
    print("\n=== utilities ===")
    # replacenan
    img = np.ones((10, 10))
    img[3:7, 3:7] = np.nan
    newimg, mask = replacenan(img, 5.0)
    check("replacenan no NaN remaining", not np.any(np.isnan(newimg)))
    check("replacenan fill value", np.all(newimg[3:7, 3:7] == 5.0))
    check("replacenan mask correct", not np.any(mask[3:7, 3:7]))

    # hysthresh
    img = np.zeros((30, 30))
    img[3:25, 15] = 5
    img[8:12, 15] = 10
    img[15:20, 15] = 10
    img[15, 15] = 20
    bw = hysthresh(img, 8, 20)
    expected = np.zeros((30, 30), dtype=bool)
    expected[15:20, 15] = True
    check("hysthresh correct result", np.array_equal(bw, expected))

    # imgnormalize
    img = np.array([[0.0, 50.0], [100.0, 200.0]])
    n = imgnormalize(img)
    check("imgnormalize min=0", abs(n.min()) < 1e-10)
    check("imgnormalize max=1", abs(n.max() - 1.0) < 1e-10)

    # histtruncate
    img = np.arange(1000, dtype=float).reshape(25, 40)
    result = histtruncate(img, 10, 10)
    check("histtruncate clipped low", result.min() >= 99)
    check("histtruncate clipped high", result.max() <= 900)


def test_bandpassmonogenic():
    print("\n=== bandpassmonogenic ===")
    np.random.seed(42)
    img = np.random.randn(32, 32)
    ph, orient, E = bandpassmonogenic(img, 4, 20, 4)
    check("bandpassmonogenic phase shape", ph.shape == img.shape)
    check("bandpassmonogenic E >= 0", np.all(E >= -1e-10))


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-validation: Python vs reference implementation")
    print("=" * 60)

    test_filtergrids()
    test_filtergrid()
    test_monogenicfilters()
    test_loggabor()
    test_lowpassfilter()
    test_bandpassfilter()
    test_perfft2()
    test_geoseries()
    test_step2line()
    test_highpassmonogenic()
    test_phasecongmono()
    test_ppdrc()
    test_phasesymmono()
    test_phasecong3()
    test_phasesym()
    test_ppdenoise()
    test_monofilt()
    test_gaborconvolve()
    test_syntheticimages()
    test_utilities()
    test_bandpassmonogenic()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
