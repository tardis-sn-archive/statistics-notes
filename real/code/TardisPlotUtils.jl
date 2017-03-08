module TardisPlotUtils
using Logging
@Logging.configure(level=INFO)

import TardisPaper.GammaIntegrand, TardisPaper.Predict, TardisPaper.Integrate, TardisPaper.Moments, TardisPaper.SmallestInterval
import Tardis
using DataFrames, Distributions, LaTeXStrings, Plots, StatPlots, StatsBase

const font = Plots.font("TeX Gyre Heros")
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)
Plots.PyPlotBackend()
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
    res ./= norm
    # but mean and σ are affected by norm
    _, mean, stderr = Integrate.simpson(Qs, res)
    (tag != "") && info(tag, ": norm=$norm, Q=$(mean)±$(stderr)")
    norm, mean, stderr
end

type GammaStatistics{T<:Real}
    q::T
    logr::T
    first::T
    second::T
    n::Int64

    function GammaStatistics(samples::Vector)
        q = sum(samples)
        logr = sum(log(samples))
        n = length(samples)
        first = q/n
        second = mapreduce(x->x^2, +, samples)/n
        new(q, logr, first, second, n)
    end
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
    normalize!(Qs, res_mle, "max. likelihood")

    # first = q/n
    # second = mapreduce(x->x^2, +, samples)/n
    res_asym_laplace = map(Q->Predict.asymptotic_by_laplace(Q, a, n, gstats.first, gstats.second), Qs)
    normalize!(Qs, res_asym_laplace, "asympt. Laplace")

    res_laplace = map(Q->Predict.by_laplace(Q, a, n, gstats.q, gstats.logr; ε=ε)[1], Qs)
    normalize!(Qs, res_laplace, "Laplace")

    res_asym_cuba = map(Q->Predict.asymptotic_by_cubature(Q, a, n, gstats.first, gstats.second; reltol=reltol)[1], Qs)
    normalize!(Qs, res_asym_cuba, "asympt. cubature")

    res_cuba = map(Q->Predict.by_cubature(Q, a, n, gstats.q, gstats.logr; reltol=reltol, ε=ε)[1], Qs)
    normalize!(Qs, res_cuba, "cubature")

    Qs, res_cuba, res_laplace, res_asym_cuba, res_asym_laplace, res_mle
end

function plot_asymptotic_single(Qs, res_cuba, res_asym_laplace; kwargs...)
    plot!(Qs, res_cuba; label=L"\sum_N", ls=:solid, leg=true, color=:red, kwargs...)
    plot!(Qs, res_asym_laplace; label=L"\int dλ", ls=:dash, color=:blue, kwargs...)
    xlabel!(L"Q")
    # ylabel!(latexstring("\$P(Q|n=$n, \\ell)\$"))
    ylabel!(L"P(Q|n,\ell)")
end

function compute_all_predictions()
    n = 10
    kwargs = Dict(:n=>n, :reltol=>1e-4, :Qs=>linspace(1e-2, 3.3, 100))
    res = Dict(n=>compute_prediction(;kwargs...))

    n = 80
    kwargs[:n] = n
    kwargs[:Qs] = linspace(0.5, 3.5, 100)
    res[n] = compute_prediction(;kwargs...)

    # n = 200
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(2.5, 8, 100)
    # res[n] = compute_prediction(;kwargs...)

    # n = 500
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(8, 16, 100)
    # res[n] = compute_prediction(;kwargs...)

    # n = 1000
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(19, 32, 100)
    # res[n] = compute_prediction(;kwargs...)

    # n = 2000
    # kwargs[:n] = n
    # kwargs[:Qs] = linspace(42, 55, 100)
    # res[n] = compute_prediction(;kwargs...)

    return res
end

function plot_asymptotic_all(res)
    plot()
    # plot the first set of curves with a label
    # but don't repeat the label on subsequent curves
    for (i, (n,x)) in enumerate(res)
        if i == 1
            plot_asymptotic_single(x...)
        else
            plot_asymptotic_single(x...; label="")
        end
    end
    annotate!([(0.58, 2.2, text(L"n=10", :center))])
    annotate!([(1.8, 1.25, text(L"n=80", :center))])

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
        # spectra = [TardisPaperPlots.analyze_spectrum(x; npackets=100, nbins=5) for x in (true, false)]
        # spectra = [TardisPaperPlots.analyze_spectrum(x; a=0.5, npackets=500, nbins=15) for x in (false, true)]
        spectra = [TardisPaperPlots.analyze_spectrum(x; a=0.5, nbins=500) for x in (false, true)]
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

function plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=2)
    # plot(Qs, res, xlabel="Q", ylabel="P(Q | n=2)", leg=true, label=L"p(N | n=2) \int d \alpha d \beta p(Q| N, \alpha, \beta) p(\alpha, \beta | n, y)")
    # plot!(Qs, resαβ, label=L"p(N | n=2) p(Q | N, \alpha_0, \beta_0)")
    # plot!(Qs, resλαβ, label=L"p(N | \lambda=2) p(Q | N, \alpha_0, \beta_0)")

    plot(Qs, resλαβ, label=L"\lambda, \alpha, \beta fixed", xlabel="Q", ylabel="P(Q | n=$nb)", leg=true)
    plot!(Qs, resαβ, label=L"\alpha, \beta fixed")
    (res[end] > 0) && plot!(Qs, res, label=L"nothing fixed")
    plot!()
end

function prepare_compare_uncertainties(kwargs...)
    K = 150

    Qs = linspace(0.001, 0.8, K)
    nb = 2
    res, resαβ, resλαβ = compare_uncertainties(Qs; nb=nb, seed=61, kwargs...)
    plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=nb)
    savepdf("comp_unc_$nb")

    Qs = linspace(0.001, 0.21, K)
    nb = 0
    # compute λ to give same δ contribution
    # Optim.optimize(x -> abs(exp(TardisPaper.GammaIntegrand.log_poisson(0, x)) - 1/sqrt(2)), 0.3, 0.4, show_trace=true)
    res, resαβ, resλαβ = compare_uncertainties(Qs; nb=nb, λ=3.465736e-01, kwargs...)
    plot_compare_uncertainties(Qs, res, resαβ, resλαβ; nb=nb)
    savepdf("comp_unc_$nb")
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
    # no tick labels
    # yticks!([])

    savepdf("tardis_input_trafo")
end

end # module
