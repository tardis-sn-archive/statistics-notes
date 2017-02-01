module TardisPaperPlots
using Logging
@Logging.configure(level=INFO)

import TardisPaper.Predict, TardisPaper.Integrate
import Tardis
using Distributions, LaTeXStrings, Plots

const font = Plots.font("TeX Gyre Heros")
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)
Plots.PyPlotBackend()

srand(16142)

"save and replot"
savepdf(fname) = begin Plots.pdf("../figures/$fname"); plot!() end

function normalize!(Qs, res, tag)
    # norm only useful from first call
    norm, _, _ = Integrate.simpson(Qs, res)
    res ./= norm
    # but mean and σ are affected by norm
    _, mean, stderr = Integrate.simpson(Qs, res)
    info(tag, ": norm=$norm, Q=$(mean)±$(stderr)")
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

    n = 200
    kwargs[:n] = n
    kwargs[:Qs] = linspace(2.5, 8, 100)
    res[n] = compute_prediction(;kwargs...)

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

function prepare_spectrum()
    # read in normalized spectrum
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

function plot_spectrum(frame)
    using StatsBase
    h=fit(Histogram, frame[:nus], weights(frame[:energies]); nbins=200)
    # off by one
    # plot(h.edges[1], h.weights)

    # works, but doesn't show the right graph
    plot(h)

    # manually subtract by one
    plot(0.01:0.02:4.3, h.weights)
end

end # module
