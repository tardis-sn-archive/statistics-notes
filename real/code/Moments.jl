module Moments

using Optim
import ForwardDiff
# chunk = ForwardDiff.Chunk{3}()
"""
Enumeration to specify which target function to create
* PosteriorParam p(\lambda, \mu, \sigma^2 | data)
"""
@enum TARGET PosteriorParam NormalAndPosterior FirstMoment SecondMoment ThirdMoment FourthMoment

"""
Create a function for Laplace approximation
"""
function targetfactory(target::TARGET, n::Int64, xmean::Float64, xsecond::Float64;
                       a::Float64=1, Q::Float64=-1.0)
    0 <= a <= 1 || error("need a in [0,1]")
    (xmean > 0) || error("nonpositive xmean $xmean")
    (xsecond > 0) || error("nonpositive xsecond $xsecond")

    """ p(\lambda | data) p(\mu, \sigma^2 | data)"""
    function logposterior_param(x::Vector)
        # all values must be positive
        any(v->v<0, x) && return -Inf
        λ, μ, σSq = x

        # gamma(λ), set β=1
        α = n-a+1
        resgamma = -lgamma(α) + (α-1)*log(λ) - λ

        # Eggers: flat prior in μ, InvGamma(σ^2|a0, b0)
        σSqn = σSq / n
        an = 0 + n/2
        bn = 0 + n/2 * (xsecond - xmean^2)
        resnormal = -1/2 * (log(2π*σSqn) + (μ-xmean)/σSqn)
        resinvgamma = an * log(bn) - lgamma(an) - (an + 1)*log(σSq) - bn / σSq

        # λ         μ           σ^2 | μ
        resgamma + resnormal + resinvgamma
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
        var = σSq + μ^2
        exponent = λ*μ^2 / (2*var)
        tmp = exp(-exponent) * sqrt(λ * var / (2π)) + 1/2 * λ * μ * (1 + erf(sqrt(exponent)))
        tmp > 0.0 || error("Negative input ", λ, μ, σSq )
        log(tmp)
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
        f = logposterior_param(x)
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
    end
    # optimizer minimizes => add minus sign at the last possible moment
    return x -> -f(x)
end

function uncertainty(n::Int64, xmean::Float64, xsecond::Float64;
                     a::Float64=1.0, Q::Float64=-1.0)
    # initial guess for optimization: slightly off to avoid
    # ERROR: Linesearch failed to converge
    init = [n * 1.0001, xmean * 1.0005, (xsecond - xmean) * 1.0005]
    # Hessian
    H = Array{Float64}(3,3)

    function integrate(moment::TARGET; Q=-1.0)
        target = targetfactory(moment, n, xmean, xsecond; a=a, Q=Q)
        res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
        print(Optim.minimizer(res))
        ForwardDiff.hessian!(H, target, Optim.minimizer(res))
        exp(tardis.laplace(-Optim.minimum(res), H))
    end

    if Q > 0.0
        return integrate(NormalAndPosterior; Q=Q)
    end

    # mean
    μ = integrate(FirstMoment)

    # second moment
    second = integrate(SecondMoment)
    σ = sqrt(second - μ^2)

    third = integrate(ThirdMoment)
    skewness = (third - 3μ*second + 2μ^3) / σ^3

    fourth = integrate(FourthMoment)
    kurtosis = (fourth - 4μ*third + 6μ^2*second - 4μ^4) / σ^4

    # excess kurtosis = kurtosis - 3
    return μ, σ, skewness, kurtosis - 3
end

end
