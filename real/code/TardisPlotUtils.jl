module TardisPlotUtils
using Logging
# @Logging.configure(level=DEBUG)
@Logging.configure(level=INFO)

import TardisPaper.GammaIntegrand, TardisPaper.Predict, TardisPaper.Integrate, TardisPaper.Moments, TardisPaper.SmallestInterval
import Tardis
using DataFrames, Distributions, LaTeXStrings, Plots, PyCall, StatPlots, StatsBase

Plots.default(grid=false)
# const font = Plots.font("TeX Gyre Heros") # Arial sans serif
const font = Plots.font("cmr10") # computer modern roman (LaTeX default)
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)
# Plots.PyPlotBackend()
# Plots.gr()

"save and replot"
savepdf(fname) = begin Plots.pdf("../figures/$fname"); plot!() end

"""# Arguments

`δcontrib`: For `N=0`, the δ function has a contribution that affects
the normalization and thus mean and std. err

"""
function normalize!(Qs, res, tag=""; δcontrib=0.0)
    # norm only useful from first call
    norm, _, _ = Integrate.simpson(Qs, res)
    norm += δcontrib
    # res ./= norm
    # but mean and σ are affected by norm
    _, mean, stderr = Integrate.simpson(Qs, res/norm)
    (tag != "") && info(tag, ": norm=$norm, Q=$(mean)±$(stderr)")
    norm, mean, stderr
end

type GammaStatistics{T<:Real}
    q::T
    logr::T
    first::T
    second::T
    n::Int64
end

function GammaStatistics{T<:Real}(samples::Vector{T})
    q = sum(samples)
    logr = sum(log(samples))
    n = length(samples)
    first = q/n
    second = mapreduce(x->x^2, +, samples)/n
    GammaStatistics{T}(q, logr, first, second, n)
end

function compute_prediction(;n=400, Qs=false, Qmin=1e-3, Qmax=2, nQ=50, α=1.5, β=60.0, a=1/2, ε=1e-3, reltol=1e-3, seed=16142)
    srand(seed)

    info("Computing for n=$n")
    dist = Gamma(α, 1/β)
    samples = rand(dist, n)
    gstats = GammaStatistics(samples)
    # q = sum(samples)
    # logr = sum(log(samples))

    if Qs === false
        meanQ = n*α/β
        Qs = linspace(Qmin*meanQ, Qmax*meanQ, nQ)
    end

    # res_known = map(Q->Predict.by_sum(Q, α, β, a, n;ε=ε)[1], Qs)
    # normalize!(Qs, res_known, "known α, β")

    dist_fit = Distributions.fit_mle(Gamma, samples)
    res_mle = map(Q->Predict.by_sum(Q, shape(dist_fit), 1/scale(dist_fit), a, n;ε=ε)[1], Qs)
    normalize!(Qs, res_mle, "max. likelihood";
               δcontrib=exp(GammaIntegrand.log_poisson_predict(0, n, a)))

    # first = q/n
    # second = mapreduce(x->x^2, +, samples)/n
    res_asym_laplace = map(Q->Predict.asymptotic_by_laplace(Q, a, n, gstats.first, gstats.second), Qs)
    normalize!(Qs, res_asym_laplace, "asympt. Laplace")

    res_laplace = map(Q->Predict.by_laplace(Q, a, n, gstats.q, gstats.logr; ε=ε)[1], Qs)
    normalize!(Qs, res_laplace, "Laplace"; δcontrib=exp(GammaIntegrand.log_poisson_predict(0, n, a)))

    res_asym_cuba = map(Q->Predict.asymptotic_by_cubature(Q, a, n, gstats.first, gstats.second; reltol=reltol)[1], Qs)
    normalize!(Qs, res_asym_cuba, "asympt. cubature")

    res_cuba = map(Q->Predict.by_cubature(Q, a, n, gstats.q, gstats.logr; reltol=reltol, ε=ε)[1], Qs)
    normalize!(Qs, res_cuba, "cubature";
               δcontrib=exp(GammaIntegrand.log_poisson_predict(0, n, a)))

    Qs, res_cuba, res_laplace, res_asym_cuba, res_asym_laplace, res_mle
end

function plot_asymptotic_single(res; kwargs...)
    Qs, cuba, laplace, asym_cuba, asym_laplace, mle = res
    plot!(Qs, cuba; label=L"\sum_N \mbox{cubature}", ls=:solid, linewidth=2, leg=true, color=:red, kwargs...)
    plot!(Qs, laplace; label=L"\sum_N \mbox{Laplace}", ls=:dash, color=:red, kwargs...)
    plot!(Qs, mle; label=L"\sum_N \mbox{MLE}", ls=:dot, color=:black, kwargs...)
    plot!(Qs, asym_cuba; label=L"\int \dd{\lambda} \mbox{cubature}", ls=:solid, color=:blue, kwargs...)
    plot!(Qs, asym_laplace; label=L"\int \dd{\lambda} \mbox{Laplace}", ls=:dash, color=:blue, kwargs...)
    xlabel!(L"Q")
    # ylabel!(latexstring("\$P(Q|n=$n, \\ell)\$"))
    ylabel!(L"P(Q|n,\ell)")
end

function compute_all_predictions()
    N = 150

    n = 10
    kwargs = Dict(:n=>n, :reltol=>1e-4, :Qs=>linspace(1e-2, 3.3, N))
    res = Dict(n=>compute_prediction(;kwargs...))

    n = 80
    kwargs[:n] = n
    kwargs[:Qs] = linspace(0.5, 3.5, N)
    res[n] = compute_prediction(;kwargs...)

    # n = 200
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(2.5, 8, N)
    # res[n] = compute_prediction(;kwargs...)

    # n = 500
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(8, 16, N)
    # res[n] = compute_prediction(;kwargs...)

    # n = 1000
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(19, 32, N)
    # res[n] = compute_prediction(;kwargs...)

    # n = 2000
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(42, 55, N)
    # res[n] = compute_prediction(;kwargs...)

    return res
end

function plot_asymptotic_all(res)
    plot()
    # plot the first set of curves with a label
    # but don't repeat the label on subsequent curves
    for (i, (n,x)) in enumerate(res)
        if i == 1
            plot_asymptotic_single(x)
        else
            plot_asymptotic_single(x; label="")
        end
    end
    annotate!([(0.7, 1.9, text(L"n=10", :center))])
    annotate!([(1.9, 1.25, text(L"n=80", :center))])

    savepdf("asymptotic")
end

function cuba_vs_laplace(res)
    lab = ("cubature", "Laplace", "asympt. cubature", "asympt. Laplace", "MLE")
    plot()
    for (k, (n,x)) in enumerate(res)
        for (i, _) = enumerate(lab)
            label = (k == 1) ? lab[i] : ""
            plot!(x[1], x[i+1], leg=true, label=label, color=i)
        end
    end
    annotate!([(0.8, 2.2, text(L"n=10", :center))])
    annotate!([(1.8, 1.25, text(L"n=80", :center))])
    annotate!([(5, 0.9, text(L"n=200", :center))])
    xlabel!(L"Q")
    ylabel!(L"P(Q)")
    savepdf("cuba_vs_laplace")
end

function tardis_raw_data()
    # read in normalized sp
    raw_data = Tardis.readdata()
    frame = Tardis.filter_positive(raw_data...)
end

"Fixed seed for reproducible spectra"
function prepare_frame(seed=1612)
    frame = tardis_raw_data()
    Tardis.transform_data!(frame)

    srand(seed)

    # update energies by drawing from Gamma distribution.
    # actual samples are only approximately from Gamma
    dist = Gamma(1.5, 1/60)
    for i in 1:length(frame[:energies])
        frame[:energies][i] = rand(dist)
    end
    frame
end

function analyze_samples(frame; npackets=typemax(Int64), nbins=20)
    npackets = min(length(frame[:nus]), npackets)
    nus = frame[:nus][1:npackets]
    energies = frame[:energies][1:npackets]
    edges = linspace(0.0, nus[end], nbins+1)
    n = Array{Int64}(nbins)
    means = zeros(nbins)
    seconds = zeros(nbins)
    variances = zeros(nbins)
    logr = zeros(nbins)
    for (i, edge) in enumerate(edges[1:end-1])
        nmin = searchsortedfirst(nus, edge)
        nmax = searchsortedlast(nus, edges[i+1])
        n[i] = nmax - nmin + 1

        if n[i] == 0
            means[i] = seconds[i] = variances[i] = logr[i] = 0.0
        else
            means[i], variances[i] = mean_and_var(energies[nmin:nmax])
            seconds[i] = mapreduce(x->x^2, +, energies[nmin:nmax]) / n[i]
            # remove Bessel correction
            variances[i] *= (n[i] - 1) / n[i]
            logr[i] = n[i] * mean(log, energies[nmin:nmax])
        end
    end

    DataFrame(left_edge=edges[1:end-1], center=(edges[2:end] + edges[1:end-1])/2,
              n=n, mean=means, second=seconds, variance=variances, logr=logr)
end

"""Estimate range in Q where probability mass is.

Rely on moments and Gaussian approximation.

# Arguments
`k`: #sigmas to include in the range
"""
function Qrange(n, Lmean, Lsecond, nb=n; k=5, nbins=100, minQ=0.0)
    # what if n small? need nb large
    # estimate moments
    μ, σ = Moments.uncertainty(n, Lmean, Lsecond, nb)

    # enlarge uncertainty to undo underestimation
    σ *= 1.5
    info("Qrange: nb=$nb, μ=$μ, σ=$σ")

    # density probably peaks at or near 0
    if μ - σ < 0
        return linspace(0, μ+2*k*σ, nbins)
    end

    # avoid getting too close to a discontinuity at zero
    # go half way between zero and next point
    s = max(minQ, μ - k*σ)
    if s == 0.0
        ΔQ = (μ + k*σ) / (nbins - 1)
        s = ΔQ/2
    end

    # take k*σ range by default
    linspace(s, μ + k*σ, nbins)
end

"""
nb < 80: sum
n < 10: cubature
n > 10: Laplace
"""
function predict_Q_bin(n, Lmean, Lsecond, logr, nb=n; a=0.5, nbins=50)
    # n=2 is the minimum to constrain two parameters
    if n < 2
        return zeros(0), zeros(0)
    end

    Qs = Qrange(n, Lmean, Lsecond, nb; nbins=nbins)

    # sum of contributing samples
    q = n * Lmean

    if nb < 80
        if n < 10
            f = Q->Predict.by_cubature(Q, a, n, q, logr, nb; reltol=1e-5)[1]
        else
            f = Q->Predict.by_laplace(Q, a, n, q, logr, nb)[1]
        end
    else
        f = Q->Predict.asymptotic_by_laplace(Q, a, n, Lmean, Lsecond, nb)
    end
    # avoid predicting for Q=0, enforce P(Q=0)=0 to avoid numerics blowing up
    # Gamma(Q=0)≡0 but it is not continuous at Q=0 if α<1.
    # Setting Q=0 thus highlights the discontinuity
    if Qs[1] == 0.0
        res = zeros(Qs)
        res[2:end] .= map(f, Qs[2:end])
    else
        res = map(f, Qs)
    end
    if nb < 3
        debug("res((min Q = $(Qs[1])) = $(res[1]), res(max Q = $(Qs[end])) = $(res[end])")
    end
    Qs, res
end

"Indices of 1σ and 2σ regions"
function analyze_bin(Qs, res; δcontrib=0.0)
    norm, mean, stderr = normalize!(Qs, res, "Analyze bin"; δcontrib=δcontrib)
    (0.98 < norm < 1.02) || warn("norm ($norm) outside of [0.98, 1.02]. Choose a better method or larger ranges for integration !?")

    one_sigma_region = SmallestInterval.connected(res, 1)
    two_sigma_region = SmallestInterval.connected(res, 2)
    one_sigma_region, two_sigma_region
    # Qs[one_sigma_region], Qs[two_sigma_region]
end

"`binposterior`: Use samples from single bin (true) or all bins (false) to determine α, β"
function analyze_spectrum(binposterior=true; a=0.5, kwargs...)
    frame = prepare_frame()
    sp = analyze_samples(frame; kwargs...)

    # add columns for the error bars
    sp[:onelo] = 0.0
    sp[:onehi] = 0.0
    sp[:twolo] = 0.0
    sp[:twohi] = 0.0
    sp[:mode]  = 0.0

    # summary statistics for all samples
    if !binposterior
        n = sum(sp[:n])
        first = sum(sp[:n] .* sp[:mean]) / n
        second = sum(sp[:n] .* sp[:second]) / n
        logr = sum(sp[:logr])
    end

    for row in eachrow(sp)
        if binposterior
            Qs, res = predict_Q_bin(row[:n], row[:mean], row[:second], row[:logr]; a=a)
        else
            Qs, res = predict_Q_bin(n, first, second, logr, row[:n]; a=a)
        end

        # something went wrong in the calculation, skip it!
        if length(Qs) == 0
            row[:mode] = row[:onelo] = row[:onehi] = row[:twolo] = row[:twohi] = NA
            continue
        end

        # upscale
        Qs, res = SmallestInterval.upscale(Qs, res, 1000)
        row[:mode] = Qs[indmax(res)]

        if row[:n] == 0
            scatter(Qs, res)
            savepdf("pred0")
        end

        one_sigma_region, two_sigma_region = analyze_bin(Qs, res; δcontrib=exp(GammaIntegrand.log_poisson_predict(0, row[:n], a)))
        row[:onelo] = Qs[one_sigma_region[1]]
        row[:onehi] = Qs[one_sigma_region[end]]
        row[:twolo] = Qs[two_sigma_region[1]]
        row[:twohi] = Qs[two_sigma_region[end]]
    end
    sp
end

function plot_spectrum(sp::DataFrame, name="spectrum"; maxQ=-1.0)

    # need empty plot
    plot(; xaxis=(L"\nu",), yaxis=(L"Q"))

    # right edges
    r = 2*sp[:center] - sp[:left_edge]

    # maximum Q value to plot, ommitting NA values
    if (maxQ < 0) maxQ = maximum(dropna(sp[:twohi])) end

    fillargs = Dict(:fillalpha=>0.4, :linealpha=>0.0)

    # plot each 95% region separately
    for (i,row) in enumerate(eachrow(sp))
        # invalid result, nothing to plot
        if row[:mode] === NA
            plot!([row[:left_edge], r[i]], zeros(2); fillcolor=:red,
                  fill_between=maxQ, fillargs...)
            continue
        end
        plot!([row[:left_edge], r[i]], [row[:twohi], row[:twohi]];
              fill_between=row[:twolo], fillcolor=:blue, fillargs...)

        y = row[:mode]
        scatter!([row[:center]], [y]; markersize=1, markercolor=:black)

        # scatter!([row[:center]], [y]; markersize=0, markercolor=:black,
        #          yerror=([y-row[:onelo]], [row[:onehi]-y])) # asymmetric yerror
        # scatter!([row[:center]], [y]; markersize=0,
        #          xerror=[row[:center]-row[:left_edge]], # symmetric xerror
        #          yerror=([y-row[:onelo]], [row[:onehi]-y])) # asymmetric yerror
    end
    ylims!(-0.02, maxQ)

    savepdf(name)
end

function compare_spectra(spectra=false)
    if spectra === false
        # spectra = [TardisPlotUtils.analyze_spectrum(x; npackets=100, nbins=5) for x in (true, false)]
        # spectra = [TardisPlotUtils.analyze_spectrum(x; a=0.5, npackets=500, nbins=15) for x in (false, true)]
        spectra = [TardisPlotUtils.analyze_spectrum(x; a=0.5, nbins=500) for x in (false, true)]
    end
    maxQ = max(maximum(dropna(spectra[1][:twohi])), maximum(dropna(spectra[2][:twohi])))
    for (sp, name) in zip(spectra, ("all", "bin"))
        plot_spectrum(sp, "spectrum_$name", maxQ=maxQ)
    end
    spectra
end

function compare_uncertainties(Qs;nb=2, λ=nb, n=nb, α=1.5, β=60.0, a=1/2, ε=1e-3, reltol=1e-5, seed=61)
    resαβ = map(Q->Predict.by_sum(Q, α, β, a, nb;ε=ε)[1], Qs)
    normalize!(Qs, resαβ, "α, β fixed"; δcontrib=exp(GammaIntegrand.log_poisson_predict(0, nb, a)))

    resλαβ = map(Q->Predict.by_sum(Q, α, β, λ;ε=ε)[1], Qs)
    normalize!(Qs, resλαβ, "λ, α, β fixed"; δcontrib=exp(GammaIntegrand.log_poisson(0, λ)))

    if nb >= 2
        dist = Gamma(α, 1/β)
        srand(seed)
        samples = rand(dist, nb)
        gstats = GammaStatistics{Float64}(samples)
        res = map(Q->Predict.by_cubature(Q, a, n, gstats.q, gstats.logr, nb;ε=ε, reltol=reltol)[1], Qs)
        normalize!(Qs, res, "nothing fixed"; δcontrib=exp(GammaIntegrand.log_poisson_predict(0, nb, a)))
    else
        res = zeros(Qs)
    end
    res, resαβ, resλαβ
end

function plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=2, upscale=200, kwargs...)
    if upscale > 0
        Qsscaled, resλαβ = SmallestInterval.upscale(Qs, resλαβ, upscale)
        _, resαβ = SmallestInterval.upscale(Qs, resαβ, upscale)
        if res[end] > 0
            _, res = SmallestInterval.upscale(Qs, res, upscale)
        end
        # avoid ArgumentError: The length of the range in dimension 1 (1000) did not equal the size of the interpolation object in that direction (5)
        Qs = Qsscaled
    end

    plot!(Qs, resλαβ; label=L"p(Q | \lambda, \alpha_0, \beta_0)", xlabel=L"Q", kwargs...)
    plot!(Qs, resαβ; label=L"p(Q | n, \alpha_0, \beta_0)", kwargs...)
    # requires Latex for text in matplotlib and \usepackage{bm}, enable in
    # ~/.config/matplotlib/matplotlibrc
    (res[end] > 0) && plot!(Qs, res; label=L"p(Q | n, \bm{\ell})", kwargs...)
end

function prepare_compare_uncertainties(kwargs...)
    # clean canvas with two plots next to each other
    plot_kwargs = Dict(:layout=>(1,3), :size=>(900, 300), :legend=>false,
                       :yticks=>nothing,
                       :grid=>false)
    plot(;plot_kwargs...)

    seed = 61
    K = 40
    Qmin = 0.001

    # position of `n=?` label in relative coordinates
    place_label(nb, subplot; xmax=0.0) = begin
        # fix ranges after autoscaling
        sub = Plots.plot!().subplots[subplot]
        x = sub.attr[:xaxis][:extrema]
        y = sub.attr[:yaxis][:extrema]
        xmax = xmax > 0.0? xmax : x.emax

        annotate!([(x.emin + 0.7*(xmax-x.emin), y.emin + 0.2*(y.emax-y.emin), text("\$n=$nb\$", :bottom))]; plot_kwargs...)
    end

    plot_kwargs[:subplot] = 1
    nb = 0
    Qs = linspace(Qmin, 0.21, K)
    # compute λ to give same δ contribution
    # Optim.optimize(x -> abs(exp(TardisPaper.GammaIntegrand.log_poisson(0, x)) - 1/sqrt(2)), 0.3, 0.4, show_trace=true)
    res, resαβ, resλαβ = compare_uncertainties(Qs; nb=nb, λ=3.465736e-01, kwargs...)
    plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=nb, plot_kwargs...)
    place_label(nb, plot_kwargs[:subplot])

    plot_kwargs[:subplot] = 2
    plot_kwargs[:legend] = true
    Qs = linspace(Qmin, 0.8, K)
    nb = 2
    res, resαβ, resλαβ = compare_uncertainties(Qs; nb=nb, seed=seed, kwargs...)
    plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=nb, plot_kwargs...)
    # zoom in on x-axis to remove long tail but need to compute it for proper normalization
    xmax = 0.6
    xlims!(Qmin, xmax, subplot=2)
    place_label(nb, plot_kwargs[:subplot]; xmax=xmax)
    plot_kwargs[:legend] = false

    plot_kwargs[:subplot] = 3
    nb = 20
    Qs = linspace(Qmin, 1.5, K)
    res, resαβ, resλαβ = compare_uncertainties(Qs; nb=nb, seed=seed, kwargs...)
    plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=nb, plot_kwargs...)
    place_label(nb, plot_kwargs[:subplot])

    savepdf("comp_unc")
end

function plot_tardis_samples()
    # first plot the raw data but sort so we have the same samples in both plots
    frame = tardis_raw_data()
    sort!(frame, cols=:nus)

    imin = 50050
    imax = imin + 1000
    nbins = 60

    layout = (1,2)

    Plots.histogram(frame[:energies][imin:imax]/1e38, nbins=nbins, yticks=nothing,
                    xlabel=L"\ell^\prime/10^{38}", normed=true, layout=layout)

    # now the transformed data, sorted again but that doesn't change anything
    Tardis.transform_data!(frame)

    samples = frame[:energies][imin:imax]
    dist_fit = Distributions.fit_mle(Gamma, samples)
    α = shape(dist_fit)
    β = 1/scale(dist_fit)
    println("α = $α, β = $β")

    Plots.histogram!(samples, xlabel=L"\ell", normed=true, nbins=nbins,
                    lab="", layout=layout, subplot=2)
    Plots.plot!(dist_fit, label=@sprintf("Gamma(%.2f, %.1f)", α, β), leg=true,
                layout=layout, subplot=2, yticks=nothing, xticks=[0.0, 0.05, 0.1])

    savepdf("tardis_input_trafo")
end

"Coordinates for a rectangle: (x0, y0) = lower left, (x1,y1) = offset from lower left to upper right corner "
function rect(x0, y0, x1,y1)
    x = [x0,x1,x1,x0,x0]
    y = [y0,y0,y1,y1,y0]
    x,y
end

"Expect `x`=frequency and `y`=luminosity. Plot spectrum + Savitzk-Golay smoothed
spectrum and zoom in on a region in an inset"
function spectrum_inset(x, y; sg_window=15, sg_order=5, nmult=5, xlim_zoom=(1.1,1.22))
    # apply Savitzky-Golay smoothing
    PyCall.@pyimport scipy.signal as sig
    sy =  sig.savgol_filter(y, sg_window, sg_order)

    # more points for smoother plotting
    finex, finey = SmallestInterval.upscale(x, sy, nmult*length(x));

    Plots.plot(x, y, leg=true, label="Monte Carlo")
    Plots.plot!(finex, finey, label="truth")
    Plots.xlabel!(L"\nu\,[10^{15} \, \mbox{Hz}]")
    Plots.ylabel!(L"\mbox{Luminosity }\,[10^{38}\, \mbox{Erg}/\mbox{s}]")

    # find min, max y in zoom box
    r = searchsortedfirst(x, xlim_zoom[1]):searchsortedlast(x, xlim_zoom[2])
    finerange = searchsortedfirst(finex, xlim_zoom[1]):searchsortedlast(finex, xlim_zoom[2])

    minx = finex[finerange.start]
    maxx = finex[finerange.stop]
    miny = 0.97 * min(minimum(finey[finerange]), minimum(y[r]))
    maxy = 1.03* max(maximum(finey[finerange]), maximum(y[r]))

    # plot zoom box
    style = Dict(:linestyle=>:dash, :color=>:black)

    rectx, recty = rect(minx, miny, maxx, maxy)
    Plots.plot!(rectx, recty; label="", style...)

    # plot arrow from zoombox toward inset
    Plots.plot!([maxx, 2.5], [ maxy, 9.2]; arrow=:arrow, label="", style...)

    # now plot the inset
    # Plots.bar!([x[r], finex[finerange]], [y[r], finey[finerange]],inset=(1, Plots.bbox(0.05, 0.25, 0.5, 0.25, :bottom, :right)), subplot=2, yticks=nothing)
    Plots.bar!(x[r], y[r], inset=(1, Plots.bbox(0.05, 0.25, 0.5, 0.25, :bottom, :right)), subplot=2, ticks=nothing, fillalpha=0.3)
    Plots.plot!( finex[finerange],  finey[finerange], subplot=2)

    savepdf("spectrum_inset")
end

end # module
