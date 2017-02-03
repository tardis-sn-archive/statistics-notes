module Moments

using ..GammaIntegrand
import ..Integrate
using Optim
import ForwardDiff
# chunk = ForwardDiff.Chunk{3}()
"""
Enumeration to specify which target function to create
* PosteriorParam p(\lambda, \mu, \sigma^2 | data)
"""
@enum TARGET PosteriorParam NormalAndPosterior FirstMoment SecondMoment ThirdMoment FourthMoment FirstMomentRaw SecondMomentRaw FirstMomentSquared Variance

"""
Create a function for Laplace approximation
"""
function targetfactory(target::TARGET, n::Int64, xmean::Float64, xsecond::Float64;
                       a::Float64=1/2, Q::Float64=-1.0)
    0 <= a <= 1 || error("need a in [0,1]")
    (xmean > 0) || error("nonpositive xmean $xmean")
    (xsecond > 0) || error("nonpositive xsecond $xsecond")

    """ p(\lambda | data) p(\mu, \sigma^2 | data)"""
    function logposterior_param(x::Vector)
        # all values must be positive
        any(v->v<0, x) && return -Inf
        λ, μ, σSq = x

        α = n-a+1

        # Eggers: flat prior in μ, InvGamma(σ^2|a0, b0)
        σSqn = σSq / n
        an = 0 + n/2
        bn = 0 + n/2 * (xsecond - xmean^2)

        # λ                  μ|σ²                         σ^2
        log_gamma(λ, α, 1) + log_normal(μ, xmean, σSqn) + log_inv_gamma(σSq, an, bn)
    end

    """N(Q | \lambda \mu, \lambda(\mu^2 + \sigma^2))"""
    function lognormal(Q::Real, x::Vector)
        λ, μ, σSq = x
        # println("Q = $Q ", λ, μ, σSq )
        var = λ*(μ^2 + σSq)
        -1/2(log(2π*var) + (Q-λ*μ)^2 / var)
    end

    """log(-d/dt f(target=0|x))"""
    function logfirst(x::Vector)
        λ, μ, σSq = x
        λ > 0 || error("Negative λ ", λ)
        μ > 0 || error("Negative μ ", μ)
        σSq > 0 || error("Negative σSq ", σSq)
        var = σSq + μ^2
        exponent = λ*μ^2 / (2*var)
        log(exp(-exponent) * sqrt(λ * var / (2π)) + 1/2 * λ * μ * (1 + erf(sqrt(exponent))))
    end

    """log(d²/dt² f(target=0|x))"""
    function logsecond(x::Vector)
        λ, μ, σSq = x
        var = σSq + μ^2
        exponent = λ*μ^2 / (2*var)
        tmp = exp(-exponent) * sqrt(λ^3 * var / (2π)) * μ + 1/2 * λ*(λ*μ^2 + var)*(1 + erf(sqrt(exponent)))
        log(tmp)
    end

    """log(-d^3/dt^3 f(target=0|x))"""
    function logthird(x::Vector)
        λ, μ, σSq = x
        var = σSq + μ^2
        exponent = λ*μ^2 / (2var)
        tmp = exp(-exponent) / sqrt(2π) * λ^(3/2) * sqrt(var) * (λ*μ^2 + 2var)
        tmp += 1/2 * λ^2 * μ * (λ*μ^2 + 3var)*(1 + erf(sqrt(exponent)))
        log(tmp)
    end

    """log(d^4/dt^4 f(target=0|x))"""
    function logfourth(x::Vector)
        λ, μ, σSq = x
        var = σSq + μ^2
        exponent = λ*μ^2 / (2*var)
        tmp = exp(-exponent) / sqrt(2π) * λ^(5/2) * μ * sqrt(var) * (λ*μ^2 + 5var)
        tmp += 1/2 * λ^2 * (λ^2*μ^4 + 6var*λ*μ^2 + 3var^2)*(1 + erf(sqrt(exponent)))
        log(tmp)
     end

    if target === PosteriorParam
        f = x -> logposterior_param(x)
    elseif target === NormalAndPosterior
        # ensure that Q explicitly given, default is Q = -1
        Q > 0 || error("Negative Q = $Q")
        f = x -> logposterior_param(x) + lognormal(Q, x)
    elseif target === FirstMoment
        f = x -> logposterior_param(x) + logfirst(x)
    elseif target === SecondMoment
        f = x -> logposterior_param(x) + logsecond(x)
    elseif target === ThirdMoment
        f = x -> logposterior_param(x) + logthird(x)
    elseif target === FourthMoment
        f = x -> logposterior_param(x) + logfourth(x)
    elseif target === FirstMomentRaw
        f = x -> logfirst(x)
    elseif target === SecondMomentRaw
        f = x -> logsecond(x)
    elseif target === FirstMomentSquared
        f = x -> logposterior_param(x) + 2*logfirst(x)
    elseif target === Variance
        f = x -> logposterior_param(x) + log(exp(logsecond(x)) - exp(2*logfirst(x)))
    end
    # optimizer minimizes => add minus sign at the last possible moment
    return x -> -f(x)
end

function uncertainty(n::Int64, xmean::Float64, xsecond::Float64;
                     a::Float64=0.5, Q::Float64=-1.0)
    # initial guess for optimization: slightly off to avoid
    # ERROR: Linesearch failed to converge
    init = [n * 1.0001, xmean * 1.0005, (xsecond - xmean^2) * 1.0005]
    # Hessian
    H = Array{Float64}(3,3)

    function integrate(moment::TARGET; Q=-1.0)
        target = targetfactory(moment, n, xmean, xsecond; a=a, Q=Q)
        res = optimize(target, init, Newton(), Optim.Options(autodiff=true))
        info(Optim.minimizer(res))
        ForwardDiff.hessian!(H, target, Optim.minimizer(res))
        exp(Integrate.by_laplace(-Optim.minimum(res), H))
    end

    if Q > 0.0
        return integrate(NormalAndPosterior; Q=Q)
    end

    # mean
    μ = integrate(FirstMoment)

    # second moment
    second = integrate(SecondMoment)
    # info("second $second")
    firstSquared = integrate(FirstMomentSquared)
    σ = sqrt(second - firstSquared)
    return μ, σ
    # lower, upper = triple_ranges(n, xmean, xsecond; k=5)
    # target = targetfactory(Variance, n, xmean, xsecond)
    # cubavar = Integrate.by_cubature(x->-target(x), lower, upper; reltol=1e-5)
    # target = targetfactory(SecondMoment, n, xmean, xsecond)
    # cuba2 = Integrate.by_cubature(x->-target(x), lower, upper; reltol=1e-5)
    # target = targetfactory(FirstMomentSquared, n, xmean, xsecond)
    # cuba1Sq = Integrate.by_cubature(x->-target(x), lower, upper; reltol=1e-5) #, abstol=0.001, maxevals=100000)
    # info("Cuba: $cuba2  $cuba1Sq")
    # info("Laplace: $second  $firstSquared")
    # return μ, σ, sqrt(cuba2[1] - cuba1Sq[1]), sqrt(integrate(Variance)), sqrt(cubavar[1])

    # TODO below wrong, doesn't take into account that (∫f)² ≠ ∫ f²
    # third = integrate(ThirdMoment)
    # skewness = (third - 3μ*second + 2μ^3) / σ^3

    # fourth = integrate(FourthMoment)
    # kurtosis = (fourth - 4μ*third + 6μ^2*second - 4μ^4) / σ^4

    # excess kurtosis = kurtosis - 3
    # return μ, σ, skewness, kurtosis - 3
end

end
