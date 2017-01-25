module TardisPaperPlots
using Logging
@Logging.configure(level=INFO)

import TardisPaper.Predict, TardisPaper.Integrate
using Distributions, LaTeXStrings, Plots

const font = Plots.font("TeX Gyre Heros")
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)
Plots.PyPlotBackend()

srand(16142)

"save and replot"
savepdf(fname) = begin Plots.pdf("../figures/$fname"); plot!() end

function compute_asymptotic(;n=400, Qs=false, Qmin=1e-3, Qmax=2, nQ=50, α=1.5, β=60.0, a=1/2, ε=1e-3, reltol=1e-3)
    dist = Gamma(α, 1/β)
    samples = rand(dist, n)
    q = sum(samples)
    logr = sum(log(samples))

    if Qs === false
        meanQ = n*α/β
        Qs = linspace(Qmin*meanQ, Qmax*meanQ, nQ)
    end

    res_cuba = map(Q->Predict.by_cubature(Q, 0.5, n, q, logr; reltol=reltol, ε=ε)[1], Qs)


    first = q/n
    second = mapreduce(x->x^2, +, samples)/n
    res_asym_laplace = map(Q->Predict.asymptotic_by_laplace(Q, a, n, first, second), Qs)

    hello(tag, norm, mean, stderr) = info(tag, ": norm=$norm, Q=$(mean)±$(stderr)")

    norm, mean, stderr = Integrate.simpson(Qs, res_cuba)
    hello("cubature", norm, mean, stderr)
    res_cuba ./= norm

    norm, mean, stderr = Integrate.simpson(Qs, res_asym_laplace)
    hello("laplace", norm, mean, stderr)
    res_asym_laplace ./= norm

    Qs, res_cuba, res_asym_laplace
end

function plot_asymptotic_single(Qs, res_cuba, res_asym_laplace; kwargs...)
    plot!(Qs, res_cuba; label=L"\sum_N", ls=:solid, leg=true, color=:red, kwargs...)
    plot!(Qs, res_asym_laplace; label=L"\int dλ", ls=:dash, color=:blue, kwargs...)
    xlabel!(L"Q")
    # ylabel!(latexstring("\$P(Q|n=$n, \\ell)\$"))
    ylabel!(L"P(Q|n,\ell)")
end

function compute_asymptotic_all()
    Qs = collect(linspace(1e-2, 3.3, 100))

    n = 10
    kwargs = Dict(:n=>n, :reltol=>1e-5, :Qs=>Qs)
    # n=>((Q, P(Q)), (annotate_x, annotate_y))
    res = Dict(n=>compute_asymptotic(;kwargs...))

    n = 80
    kwargs[:n] = n
    res[n] = compute_asymptotic(;kwargs...)
    res
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

end # module
