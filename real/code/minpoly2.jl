module MinPoly

using NLopt
using Polynomials

type OptData
    p::Poly
    dp::Poly
    count::Unsigned

    function OptData(θ::Vector)
        p = Poly(θ)
        dp = polyder(p)
        new(p, dp, 0)
    end
end

function update_poly!(popt::OptData, θ::Vector)
    # first the actual polynomial
    # then its derivative
    for i in eachindex(popt.p)
        popt.p[i] = θ[i]
        if i > 1
            popt.dp[i]
        end
    end
end

function factory(θ)
    popt = OptData(θ)
    myfunc = function(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1] = polyval(popt.dp, x[1])
        end
        popt.count += 1
        polyval(popt.p, x[1])
    end
    popt, myfunc
end

function create_optimizer(θ::Vector, xmin::Real, xmax::Real)
    opt = Opt(:LD_VAR2, 1)
    xtol_rel!(opt, 1e-16)
    lower_bounds!(opt, [xmin])
    upper_bounds!(opt, [xmax])
    popt, f = factory(θ)
    min_objective!(opt, f)
    return opt, popt
end

function test()
    θ = [1, 0.0, -2, 0]
    opt, popt = create_optimizer(θ, 0.0, 1.0)
    @time minf, minx, ret = optimize(opt, [0.5])
    println("got $minf at $minx after $(popt.count) iterations (return $ret)")
end

end # module MinPoly

MinPoly.test()

# Local Variables:
# compile-command:"julia minpoly2.jl"
# End:
