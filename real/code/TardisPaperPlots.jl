module TardisPaperPlots
using Logging
@Logging.configure(level=INFO)

import TardisPaper.Predict, TardisPaper.Integrate, TardisPaper.Moments, TardisPaper.SmallestInterval
import Tardis
using DataFrames, Distributions, LaTeXStrings, Plots, StatPlots, StatsBase

const font = Plots.font("TeX Gyre Heros")
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)
Plots.PyPlotBackend()
# Plots.gr()

srand(16142)

"save and replot"
savepdf(fname) = begin Plots.pdf("../figures/$fname"); plot!() end

function normalize!(Qs, res, tag="")
    # norm only useful from first call
    norm, _, _ = Integrate.simpson(Qs, res)
    res ./= norm
    # but mean and σ are affected by norm
    _, mean, stderr = Integrate.simpson(Qs, res)
    (tag != "") && info(tag, ": norm=$norm, Q=$(mean)±$(stderr)")
    norm, mean, stderr
end

function compute_prediction(;n=400, Qs=false, Qmin=1e-3, Qmax=2, nQ=50, α=1.5, β=60.0, a=1/2, ε=1e-3, reltol=1e-3)
    info("Computing for n=$n")
    dist = Gamma(α, 1/β)
    samples = rand(dist, n)
    q = sum(samples)
    logr = sum(log(samples))

    if Qs === false
        meanQ = n*α/β
        Qs = linspace(Qmin*meanQ, Qmax*meanQ, nQ)
    end

    # res_known = map(Q->Predict.by_sum(Q, α, β, a, n;ε=ε)[1], Qs)
    # normalize!(Qs, res_known, "known α, β")

    dist_fit = Distributions.fit_mle(Gamma, samples)
    res_mle = map(Q->Predict.by_sum(Q, shape(dist_fit), 1/scale(dist_fit), a, n;ε=ε)[1], Qs)
    normalize!(Qs, res_mle, "max. likelihood")

    first = q/n
    second = mapreduce(x->x^2, +, samples)/n
    res_asym_laplace = map(Q->Predict.asymptotic_by_laplace(Q, a, n, first, second), Qs)
    normalize!(Qs, res_asym_laplace, "asympt. Laplace")

    res_laplace = map(Q->Predict.by_laplace(Q, a, n, q, logr; ε=ε)[1], Qs)
    normalize!(Qs, res_laplace, "Laplace")

    res_asym_cuba = map(Q->Predict.asymptotic_by_cubature(Q, a, n, first, second; reltol=reltol)[1], Qs)
    normalize!(Qs, res_asym_cuba, "asympt. cubature")

    res_cuba = map(Q->Predict.by_cubature(Q, a, n, q, logr; reltol=reltol, ε=ε)[1], Qs)
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

function prepare_frame()
    # read in normalized sp
    raw_data = Tardis.readdata()
    frame = Tardis.filter_positive(raw_data...)
    Tardis.transform_data!(frame)

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
    variances = zeros(nbins)
    logr = zeros(nbins)
    for (i, edge) in enumerate(edges[1:end-1])
        nmin = searchsortedfirst(nus, edge)
        nmax = searchsortedlast(nus, edges[i+1])
        n[i] = nmax - nmin + 1
        # println("$nmin, $nmax")
        if n[i] == 0
            means[i] = variances[i] = logr[i] = 0.0
        else
            means[i], variances[i] = mean_and_var(energies[nmin:nmax])
            # remove Bessel correction
            variances[i] *= (n[i] - 1) / n[i]
            logr[i] = n[i] * mean(log, energies[nmin:nmax])
        end
    end

    # indices of right-most element in each bin
    # indr = cumsum(n)

    return DataFrame(left_edge=edges[1:end-1],
                     center=(edges[2:end] + edges[1:end-1])/2,
                     n=n, mean=means, variance=variances, logr=logr)

end

function Qrange(n, Qmean, Qvar; k=5, nbins=100, minQ=0.0)
    # estimate moments
    μ, σ = Moments.uncertainty(n, Qmean, Qvar+Qmean^2)

    # TODO what if n small? n == 1?
    # use MLE result with enlargement or some other

    # enlarge uncertainty to undo underestimation
    σ *= 1.5
    info("$μ, $σ")

    # avoid getting too close to a discontinuity at zero
    # go half way between zero and next point
    s = max(minQ, μ - k*σ)
    if s == 0.0
        ΔQ = (μ + k*σ) / (nbins - 1)
        s = ΔQ/2
    end

    # take k*σ range by default
    res = linspace(s, μ + k*σ, nbins)
end

"""
n < 10: cubature sum
10 <= n <= 80: Laplace sum
80 < n: asymptotic Laplace
"""
function predict_Q_bin(n, Qmean, Qvar, logr; a=0.5, nbins=50)
    (n == 0) && error("hit empty bin")
    Qs = Qrange(n, Qmean, Qvar; nbins=nbins)
    q = n*Qmean
    n > 8 || error("Hit bin with 10>n=$n packets. Method unstable!")
    if n < 10
        f = Q->Predict.by_cubature(Q, a, n, q, logr; reltol=1e-5)[1]
    elseif 10 <= n < 80
        f = Q->Predict.by_laplace(Q, a, n, q, logr)[1]
    elseif n > 80
        second = Qvar + Qmean^2
        f = Q->Predict.asymptotic_by_laplace(Q, a, n, Qmean, second)
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
    Qs, res
end

"Indices of 1σ and 2σ regions"
function analyze_bin(Qs, res)
    norm, mean, stderr = normalize!(Qs, res, "Analyze bin")
    (0.95 < norm < 1.05) || warn("norm ($norm) outside of [0.9, 1.1]. Choose a better integration method !?")

    one_sigma_region = SmallestInterval.connected(res, 1)
    two_sigma_region = SmallestInterval.connected(res, 2)
    one_sigma_region, two_sigma_region
    # Qs[one_sigma_region], Qs[two_sigma_region]
end

# overwrite from StatsBase
@recipe function f(h::StatsBase.Histogram)
    seriestype := :path
    linetype := :steppost
    linecolor --> :blue
    h.edges[1], h.weights
end

function analyze_spectrum(;kwargs...)
    frame = prepare_frame()
    sp = analyze_samples(frame; kwargs...)
    println(sp)

    # add columns for the error bars
    sp[:onelo] = 0.0
    sp[:onehi] = 0.0
    sp[:twolo] = 0.0
    sp[:twohi] = 0.0
    sp[:mode]  = 0.0

    for row in eachrow(sp)
        Qs, res = predict_Q_bin(row[:n], row[:mean], row[:variance], row[:logr])
        # upscale
        Qs, res = SmallestInterval.upscale(Qs, res, 1000)
        row[:mode] = Qs[indmax(res)]

        one_sigma_region, two_sigma_region = analyze_bin(Qs, res)
        row[:onelo] = Qs[one_sigma_region[1]]
        row[:onehi] = Qs[one_sigma_region[end]]
        row[:twolo] = Qs[two_sigma_region[1]]
        row[:twohi] = Qs[two_sigma_region[end]]
    end
    sp
end

function plot_spectrum(sp::DataFrame)

    y = sp[:mode]

    # need empty plot
    plot(; xaxis=(L"\nu",), yaxis=(L"Q"))

    # right edges
    r = 2*sp[:center] - sp[:left_edge]

    # plot each 95% region separately
    for (i,row) in enumerate(eachrow(sp))
        plot!([row[:left_edge], r[i]], [row[:twohi], row[:twohi]];
              fill_between=row[:twolo], fillalpha=0.4, fillcolor=:green, linealpha=0.0)
    end

    scatter!(sp[:center], y; markersize=2,
            xerror=sp[:center]-sp[:left_edge], # symmetric xerror
            yerror=(y-sp[:onelo], sp[:onehi]-y)) # asymmetric yerror

    # y = sp[:n] .* sp[:mean]
    # kwargs = Dict(:fillalpha=>0.4, :fillcolor=>:green, :color=>:red, :linetype=>:steppost)
    # # kwargs[:primary] = false
    # settings = Dict()
    # settings[:twohi] = settings[:twolo] = Dict(:color=>:blue)
    # settings[:onehi] = settings[:onelo] = Dict(:color=>:green)
    # for x in (:twohi, :twolo, :onehi, :onelo)
    #     plot!(sp[:edge], sp[x]; merge(kwargs, settings[x])...)
    # end

    savepdf("spectrum")
    nothing
end

end # module
