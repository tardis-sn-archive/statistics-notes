module PaperShortPlotUtils

using Plots
using Distributions
using LaTeXStrings

# set up output
const font = Plots.font("cmr10") # computer modern roman (LaTeX default)
Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)

"save and replot"
savepdf(fname) = begin Plots.pdf("../figures/$fname"); plot!() end

const a = 1/2

function decorate()
    plot(; xlabel=L"Q_{\infty}", ylabel=L"p(Q_{\infty} | n, \bm{\ell})", yticks=nothing, grid=false, legend=true)
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

function plot_λvariable(nps=[100,500, 5000], trueQ=1.0)
    decorate()
    for np in nps
        λ = 0.01 * np
        n = Int64(floor(λ))
        meanℓ = trueQ / λ
        Δ = 5*sqrt(n-a+1)*meanℓ
        plot!(linspace(max(0.0, trueQ - Δ) , trueQ + Δ, 400), Q0 -> pdf(Gamma(n-a+1, 1), Q0/meanℓ)/meanℓ, label="\$ \\lambda = n = $n \$")
    end
    vline!([trueQ], color=:black, ls=:dash, lab="")
    xlims!(0, 3.5)
    savepdf("contraction")
end

end # module
