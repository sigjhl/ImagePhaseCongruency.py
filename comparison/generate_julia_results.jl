#=
Generate reference outputs from Julia ImagePhaseCongruency for comparison
with the Python transpilation.

Usage:
    julia --project=.. generate_julia_results.jl [output_dir]

Requires: ImagePhaseCongruency, NPZ (for saving .npz files)

Install dependencies:
    ] add ImagePhaseCongruency NPZ

The script generates deterministic test data and saves all function outputs
as .npy files that the Python comparison script can load and diff.
=#

using ImagePhaseCongruency
using NPZ
using Random
using FFTW
using Statistics

outdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "julia_results")
mkpath(outdir)

save(name, arr) = npzwrite(joinpath(outdir, name * ".npy"), arr)
save(name, val::Number) = npzwrite(joinpath(outdir, name * ".npy"), [val])

println("Saving Julia reference outputs to: $outdir")
println("=" ^ 60)

# ====================================================================
# Deterministic test image: a 64x64 step2line
# ====================================================================
img64 = step2line(64, nscales=10)
save("step2line_64_10", img64)

img128 = step2line(128)
save("step2line_128_default", img128)

# ====================================================================
# Deterministic random image (use MersenneTwister for reproducibility)
# ====================================================================
rng = MersenneTwister(42)
randimg = randn(rng, 64, 64)
save("randimg_64", randimg)

# For 32x32 tests
rng32 = MersenneTwister(42)
randimg32 = randn(rng32, 32, 32)
save("randimg_32", randimg32)

# ====================================================================
# 1. FREQUENCY FILTERS
# ====================================================================
println("\n--- frequencyfilt ---")

# filtergrids
for (rows, cols) in [(64, 64), (63, 63), (101, 200), (128, 100)]
    f, fx, fy = filtergrids(rows, cols)
    save("filtergrids_$(rows)x$(cols)_f", f)
    save("filtergrids_$(rows)x$(cols)_fx", fx)
    save("filtergrids_$(rows)x$(cols)_fy", fy)
    println("  filtergrids($rows, $cols)")
end

# filtergrid
for (rows, cols) in [(64, 64), (63, 63), (101, 200)]
    f = filtergrid(rows, cols)
    save("filtergrid_$(rows)x$(cols)", f)
    println("  filtergrid($rows, $cols)")
end

# monogenicfilters
for (rows, cols) in [(64, 64), (101, 200)]
    H1, H2, f = monogenicfilters(rows, cols)
    save("monofilters_$(rows)x$(cols)_H1_real", real.(H1))
    save("monofilters_$(rows)x$(cols)_H1_imag", imag.(H1))
    save("monofilters_$(rows)x$(cols)_H2_real", real.(H2))
    save("monofilters_$(rows)x$(cols)_H2_imag", imag.(H2))
    save("monofilters_$(rows)x$(cols)_f", f)
    println("  monogenicfilters($rows, $cols)")
end

# packedmonogenicfilters
for (rows, cols) in [(64, 64), (101, 200)]
    H, f = packedmonogenicfilters(rows, cols)
    save("packedmono_$(rows)x$(cols)_H_real", real.(H))
    save("packedmono_$(rows)x$(cols)_H_imag", imag.(H))
    save("packedmono_$(rows)x$(cols)_f", f)
    println("  packedmonogenicfilters($rows, $cols)")
end

# lowpassfilter
for (rows, cols, cutoff, n) in [(64, 64, 0.25, 2), (101, 200, 0.4, 15)]
    f = lowpassfilter((rows, cols), cutoff, n)
    save("lowpass_$(rows)x$(cols)_$(cutoff)_$(n)", f)
    println("  lowpassfilter(($rows, $cols), $cutoff, $n)")
end

# highpassfilter
f = highpassfilter((64, 64), 0.25, 2)
save("highpass_64x64_0.25_2", f)
println("  highpassfilter((64, 64), 0.25, 2)")

# bandpassfilter
f = bandpassfilter((64, 64), 0.1, 0.3, 2)
save("bandpass_64x64_0.1_0.3_2", f)
println("  bandpassfilter((64, 64), 0.1, 0.3, 2)")

# highboostfilter
f = highboostfilter((64, 64), 0.25, 2, 2.0)
save("highboost_64x64_0.25_2_2.0", f)
println("  highboostfilter((64, 64), 0.25, 2, 2.0)")

# loggabor (scalar)
for (fval, fo, sigma) in [(0.0, 0.3, 0.55), (0.1, 0.3, 0.55), (0.3, 0.3, 0.55), (0.5, 0.3, 0.55)]
    v = loggabor(fval, fo, sigma)
    save("loggabor_$(fval)_$(fo)_$(sigma)", v)
    println("  loggabor($fval, $fo, $sigma) = $v")
end

# gridangles
f, fx, fy = filtergrids(64, 64)
sintheta, costheta = gridangles(f, fx, fy)
save("gridangles_64x64_sin", sintheta)
save("gridangles_64x64_cos", costheta)
println("  gridangles(64, 64)")

# cosineangularfilter
fltr = cosineangularfilter(pi/4, pi/3, sintheta, costheta)
save("cosangfilt_64x64", fltr)
println("  cosineangularfilter(pi/4, pi/3, ...)")

# gaussianangularfilter
fltr = gaussianangularfilter(pi/4, 0.4, sintheta, costheta)
save("gaussangfilt_64x64", fltr)
println("  gaussianangularfilter(pi/4, 0.4, ...)")

# perfft2
P, S, p, s = perfft2(randimg)
save("perfft2_P_real", real.(P))
save("perfft2_P_imag", imag.(P))
save("perfft2_S_real", real.(S))
save("perfft2_S_imag", imag.(S))
save("perfft2_p", p)
save("perfft2_s", s)
println("  perfft2(randimg)")

# geoseries
s1 = geoseries(0.5, 2.0, 4)
save("geoseries_0.5_2_4", collect(s1))
s2 = geoseries((0.5, 4.0), 4)
save("geoseries_tuple_4", collect(s2))
println("  geoseries")

# ====================================================================
# 2. SYNTHETIC IMAGES
# ====================================================================
println("\n--- syntheticimages ---")

save("step2line_custom", step2line(128, nscales=20, ampexponent=-1.5, ncycles=3.0, phasecycles=0.5))
println("  step2line(128, custom)")

save("circsine_64", circsine(64, wavelength=20, nscales=10))
println("  circsine(64)")

save("circsine_65_offset", circsine(65, wavelength=40, nscales=1, offset=pi/2))
println("  circsine(65, offset=pi/2)")

save("starsine_64", starsine(64, ncycles=10, nscales=10))
println("  starsine(64)")

save("starsine_65_offset", starsine(65, ncycles=10, nscales=1, offset=0.0))
println("  starsine(65, offset=0)")

# noiseonf — uses random, so save a specific random stream version
Random.seed!(42)
save("noiseonf_64_1.5", noiseonf(64, 1.5))
println("  noiseonf(64, 1.5)")

# quantizephase
save("quantizephase_4", quantizephase(randimg, 4))
println("  quantizephase(randimg, 4)")

# nophase — uses random, skip exact comparison but save for shape/range check
Random.seed!(123)
save("nophase_randimg", nophase(randimg))
println("  nophase(randimg)")

# ====================================================================
# 3. UTILITIES
# ====================================================================
println("\n--- utilities ---")

# histtruncate
save("histtruncate_10_10", histtruncate(randimg, 10.0, 10.0))
println("  histtruncate(randimg, 10, 10)")

save("histtruncate_5", histtruncate(randimg, 5.0))
println("  histtruncate(randimg, 5)")

# imgnormalize
save("imgnormalize_01", imgnormalize(randimg))
println("  imgnormalize(randimg)")

save("imgnormalize_mv", imgnormalize(randimg, 5.0, 2.0))
println("  imgnormalize(randimg, 5, 2)")

# replacenan
nanimg = copy(randimg)
nanimg[10:20, 10:20] .= NaN
newimg, mask = replacenan(nanimg, 99.0)
save("replacenan_img", newimg)
save("replacenan_mask", Float64.(mask))
println("  replacenan")

# hysthresh
htimg = zeros(30, 30)
htimg[3:25, 15] .= 5.0
htimg[8:12, 15] .= 10.0
htimg[15:20, 15] .= 10.0
htimg[15, 15] = 20.0
bw = hysthresh(htimg, 8.0, 20.0)
save("hysthresh_result", Float64.(bw))
println("  hysthresh")

# fillnan
nanimg2 = copy(Float64.(randimg[1:20, 1:20]))
nanimg2[5:10, 5:10] .= NaN
try
    newimg2, mask2 = fillnan(nanimg2)
    save("fillnan_img", newimg2)
    save("fillnan_mask", Float64.(mask2))
    println("  fillnan")
catch err
    println("  fillnan (skipped due to Julia runtime error: ", err, ")")
end

# ====================================================================
# 4. PHASE CONGRUENCY FUNCTIONS
# ====================================================================
println("\n--- phasecongruency ---")

# highpassmonogenic
ph, orient, E = highpassmonogenic(randimg32, 20.0, 4)
save("highpassmono_phase", ph)
save("highpassmono_orient", orient)
save("highpassmono_E", E)
println("  highpassmonogenic(32x32, 20, 4)")

# bandpassmonogenic
ph, orient, E = bandpassmonogenic(randimg32, 4.0, 20.0, 4)
save("bandpassmono_phase", ph)
save("bandpassmono_orient", orient)
save("bandpassmono_E", E)
println("  bandpassmonogenic(32x32, 4, 20, 4)")

# ppdrc single
dimg = ppdrc(randimg32 .* 100, 20.0, clip=0.01, n=2)
save("ppdrc_single", dimg)
println("  ppdrc(32x32*100, 20)")

# ppdrc multi
dimgs = ppdrc(randimg32 .* 100, geoseries((10.0, 30.0), 3), clip=0.01, n=2)
for (i, d) in enumerate(dimgs)
    save("ppdrc_multi_$i", d)
end
save("ppdrc_multi_count", length(dimgs))
println("  ppdrc(32x32*100, geoseries)")

# phasecongmono
PC, or_pcm, ft, T_pcm = phasecongmono(randimg32, nscale=3, minwavelength=3, mult=2.0,
                                         sigmaonf=0.55, k=3.0, noisemethod=-1.0,
                                         cutoff=0.5, g=10.0, deviationgain=1.5)
save("phasecongmono_PC", PC)
save("phasecongmono_or", or_pcm)
save("phasecongmono_ft", ft)
save("phasecongmono_T", T_pcm)
println("  phasecongmono(32x32)")

# phasesymmono
phSym, symE, T_psm = phasesymmono(randimg32, nscale=3, minwavelength=3, mult=2.0,
                                     sigmaonf=0.55, k=2.0, polarity=0, noisemethod=-1.0)
save("phasesymmono_phSym", phSym)
save("phasesymmono_symE", symE)
save("phasesymmono_T", T_psm)
println("  phasesymmono(32x32)")

# phasecong3
M, m, or_pc3, featType, EO, T_pc3 = phasecong3(randimg32, nscale=3, norient=4,
                                                   minwavelength=3, mult=2.0,
                                                   sigmaonf=0.55, k=2.0,
                                                   cutoff=0.5, g=10.0, noisemethod=-1.0)
# Use distinct names that do not differ only by case to avoid collisions on
# case-insensitive filesystems.
save("phasecong3_Mmax", M)
save("phasecong3_mmin", m)
save("phasecong3_or", or_pc3)
save("phasecong3_ft", featType)
save("phasecong3_T", T_pc3)
# Save a few EO slices
save("phasecong3_EO_1_1_real", real.(EO[1,1]))
save("phasecong3_EO_1_1_imag", imag.(EO[1,1]))
save("phasecong3_EO_2_3_real", real.(EO[2,3]))
save("phasecong3_EO_2_3_imag", imag.(EO[2,3]))
println("  phasecong3(32x32)")

# phasesym
phSym2, orient2, totE, T_ps = phasesym(randimg32, nscale=3, norient=4,
                                          minwavelength=3, mult=2.0, sigmaonf=0.55,
                                          k=2.0, polarity=0, noisemethod=-1.0)
save("phasesym_phSym", phSym2)
save("phasesym_orient", orient2)
save("phasesym_totE", totE)
save("phasesym_T", T_ps)
println("  phasesym(32x32)")

# ppdenoise
clean = ppdenoise(randimg32, nscale=3, norient=4, mult=2.5,
                    minwavelength=2, sigmaonf=0.55,
                    dthetaonsigma=1.0, k=3.0, softness=1.0)
save("ppdenoise_clean", clean)
println("  ppdenoise(32x32)")

# monofilt
f_mf, h1f_mf, h2f_mf, A_mf, theta_mf, psi_mf = monofilt(randimg32, 3, 3, 2.0, 0.55, false)
for s in 1:3
    save("monofilt_f_$s", f_mf[s])
    save("monofilt_h1f_$s", h1f_mf[s])
    save("monofilt_h2f_$s", h2f_mf[s])
    save("monofilt_A_$s", A_mf[s])
    save("monofilt_theta_$s", theta_mf[s])
    save("monofilt_psi_$s", psi_mf[s])
end
println("  monofilt(32x32)")

# gaborconvolve
EO_gc, BP_gc = gaborconvolve(randimg32, 3, 4, 3, 2.0, 0.55, 1.3, 0)
for s in 1:3
    save("gabor_BP_$s", BP_gc[s])
    for o in 1:4
        save("gabor_EO_$(s)_$(o)_real", real.(EO_gc[s,o]))
        save("gabor_EO_$(s)_$(o)_imag", imag.(EO_gc[s,o]))
    end
end
println("  gaborconvolve(32x32)")

println("\n" * "=" ^ 60)
println("Done! Saved all reference outputs to: $outdir")
println("Now run: python compare_with_julia.py $outdir")
