module PaperShortPlotUtils

using Integrate
using Plots
using Distributions
using LaTeXStrings

# set up output
const font = Plots.font("cmr10") # computer modern roman (LaTeX default)
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)

const mylines = Dict(1=>Dict(:color=>"#d62728"),
                     2=>Dict(:color=>"#bcbd22"),
                     3=>Dict(:color=>"#9467bd"))


"save and replot"
savepdf(fname) = begin Plots.pdf("../figures/$fname"); plot!() end

const a = 1/2

function decorate()
    plot(; xlabel=L"L_{\infty}", ylabel=L"p(L_{\infty} | n, \langle\ell\rangle)",
         yticks=nothing, grid=false, legend=true)
end

function plot_convolution()

    plot(; xlabel=L"L_{\infty}",
         yticks=nothing, grid=false, legend=true)
    meanℓ = 0.2
    n = 6
    Lmin = 1e-7; Lmax = 9
    Lrange = linspace(Lmin, Lmax, 400)
    f(Q0) = pdf(Gamma(n-a+1, 1), Q0/meanℓ)/meanℓ
    plot!(Lrange, f; label=L"p(L_{\infty} | \bm{\theta})", mylines[3]...)

    Lobs = 1.8
    g(Q0, Q=Lobs) = pdf(Normal(Q, 0.3), Q0)
    plot!(Lrange, g; label=L"\mathcal{N}(L_{\rm obs} | L_{\infty}, \sigma^2_{\rm obs})",  mylines[2]...)
    vline!([Lobs]; ls=:dash, label="", mylines[2]...)

    # print(Integrate.by_cubature(Q0 -> log(f(Q0)) + log(g(Q0, 3)), Lmin, Lmax))
    pl = map(Q -> Integrate.by_cubature(Q0 -> log(f(Q0)) + log(g(Q, Q0)), Lmin, Lmax)[1], Lrange)
    norm, mean, std = Integrate.simpson(Lrange, pl)
    print("normalization: ", norm, ", mean: ", mean, ", std: " , std)
    plot!(Lrange, pl;
          label=L"p(L_{\rm obs} | \bm{\theta})", linewidth=2, mylines[1]...)

    xlims!(Lmin, 4)

    savepdf("convolution")

    nothing
end

# asumme phi fixed
function plot_λfixed(ns=[0,1,2,5], meanℓ=1.0)
    decorate()
    for n in ns
        # n = rand(Poisson(λ))
        # k = kfactor*n
        # meanℓ = trueQ / n
        plot!(linspace(1e-7, 12, 400), Q0 -> pdf(Gamma(n-a+1, 1), Q0/meanℓ)/meanℓ, label="\$n=$n\$")
    end
    # mode for n=1 is large y value to be shown
    ylims!(0, 1.05*pdf(Gamma(2-a, 1), (1-a))/meanℓ)
    savepdf("fixed_lambda")
end

function plot_λvariable(input=[1, 10, 100], trueQ=1.0)
    decorate()
    for (j, λ) in enumerate(input)
        linestyles = [:solid, :dash, :dot]
        incr = (λ == 1) ? [0, 1, -1] : [0]
        lab = "\$ \\lambda="
        if λ == 1
            lab *= "$λ, \\, n=0,1,2"
        else
            lab *= "n=$λ"
        end
        lab *= "\$"
        for (i, ls) in zip(incr, linestyles)
            n = Int64(floor(λ)) + i
            meanℓ = trueQ / λ
            Δ = 5*sqrt(n-a+1)*meanℓ
            plot!(linspace(max(1e-3, trueQ - Δ) , trueQ + Δ, 400), Q0 -> pdf(Gamma(n-a+1, 1), Q0/meanℓ)/meanℓ; ls=ls,
                  label=(i == 0)? lab : "",
                  mylines[j]...
                  )
        end
    end
    vline!([trueQ], color=:black, ls=:dash, lab="")
    xlims!(0, 3.1)
    ylims!(0, 4.2)
    annotate!([(0.25, 1.7, text(L"n=0", :center))])
    annotate!([(0.17, 0.53, text(L"n=1", :center))])
    annotate!([(2.5, 0.4, text(L"n=2", :center))])

    # ylabel!(L"p(Q_{\infty} | n=0.01 n_p -1, \langle\ell\rangle)")
    savepdf("contraction")
end

end # module
