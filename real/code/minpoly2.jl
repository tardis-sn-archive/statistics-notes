using NLopt
using Polynomials

type PolyOpt
    p::Poly
    dp::Poly
    count::Unsigned

    function PolyOpt(θ::Vector)
        p = Poly(θ)
        dp = polyder(p)
        new(p, dp, 0)
    end
end

function factory(θ)
    popt = PolyOpt(θ)
    myfunc = function(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = polyval(popt.dp, x[1])
        end
        popt.count += 1
        polyval(popt.p, x[1])
    end
    popt, myfunc
end

θ = [1, -2, 0, 1.34]
opt = Opt(:LD_VAR2, 1)
xtol_rel!(opt, 1e-16)
lower_bounds!(opt, [0.0])
upper_bounds!(opt, [1.0])
xinit = [0.5]
popt, f = factory(θ)
println(f([0.1], [0.3]))
min_objective!(opt, f)
@time minf, minx, ret = optimize(opt, xinit)
println("got $minf at $minx after $(popt.count) iterations (return $ret)")

# Local Variables:
# compile-command:"julia minpoly2.jl"
# End:
