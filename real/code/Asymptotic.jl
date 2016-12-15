module Asymptotic

using ..Tardis
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
function targetfactory(target::TARGET, n::Int64, xmean::Float64, xsumsq::Float64;
                       a::Float64=1, Q::Float64=-1.0)
    0 <= a <= 1 || error("need a in [0,1]")
    (xmean > 0) || error("nonpositive xmean $xmean")
    (xsumsq > 0) || error("nonpositive xsumsq $xsumsq")

    """ p(\lambda | data) p(\mu, \sigma^2 | data)"""
    function logposterior_param(x::Vector)
        # all values must be positive
        any(v->v<0, x) && return -Inf
        λ, μ, σSq = x

        # gamma(λ), set β=1
        α = n-a+1
        resgamma = - lgamma(α) + (α-1)*log(λ) - λ
        ## println(resgamma)

        # normal-inversegamma
        # d = ConjugatePriors.NormalInverseGamma(xmean, n, n/2, xsumsq/2)
        # resnormal = -1/2 * (log(2*π*σSq / d.v0) + (μ-d.mu)^2 * d.v0 / σSq)
        # resinvgamma = -lgamma(d.shape) + d.shape * log(d.scale) - (d.shape + 1) * log(σSq) - d.scale / σSq

        # Murphy: inverse Gamma (188), (196)-(200), (280)
        # Vn = n
        # mn = n*xmean
        # an = n/2
        # bn = 1/2*xsumsq # - mn^2 / Vn)
        # resnormal = -1/2(log(2π*Vn*σSq) + (μ - mn)^2/(Vn*σSq))
        # resinvgamma = an*log(bn) - lgamma(an) - (an+1)*log(σSq) - bn/σSq

        # Murphy: scaled inverse χ^2
        # (132), (141)-(144), (285) take uninformative hyperparameters
        # μ0 = κ0 = ν0 = 0
        # σSq0 = 1 # irrelevant but positive
        # κn = κ0 + n
        # νn = ν0 + n
        # μn = (κ0*μ0 + n*xmean) / κn
        # σSqn = (ν0*σSq0 + xsumsq + n*κ0 / (κ0 + n)*(μ0 - xmean)) / νn
        # resnormal = -1/2 * (log(2π*σSq/κn) + (μ - μn)^2/(σSq/κn))
        # resinvgamma = -lgamma(νn/2) + νn/2*log(νn*σSqn/2) - (νn/2+1)*log(σSq) - νn*σSqn/(2*σSq)

        # Eggers: flat prior in μ, InvGamma(σ^2|a0, b0)
        σSqn = σSq / n
        an = 0 + n/2
        bn = 0 + n/2 * (xsumsq)
        resnormal = -1/2 * (log(2π*σSqn) + (μ-xmean)/σSqn)
        resinvgamma = an * log(bn) - lgamma(an) - (an + 1)*log(σSq) - bn / σSq

        ## println(resinvgamma," ", resnormal)
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
        tmp = exp(-exponent) * sqrt(λ * var / (2 * π)) + 1/2 * λ * μ * (1 + erf(sqrt(exponent)))
        tmp > 0.0 || error(λ, μ, σSq )
        log(tmp)
    end

    """log(d²/dt² f(target=0|x))"""
    function logsecond(x::Vector)
        λ, μ, σSq = x
        var = σSq + μ^2
        exponent = λ*μ^2 / (2*var)
        tmp = exp(-exponent) * sqrt(2 / π * λ^3 * var) * μ + 1/2 * λ*(λ*μ^2 + var)*(1 + erf(sqrt(exponent)))
        log(1/2 * tmp)
    end

    """log(-d^3/dt^3 f(target=0|x))"""
    function logthird(x::Vector)
        λ, μ, σSq = x
        var = σSq + μ^2
        exponent = λ*μ^2 / 2var
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
        tmp += 1/2 * λ^2 * (λ^2*μ^4 + 6λ*μ^2 + 3var)*(1 + erf(sqrt(exponent)))
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

function uncertainty(n::Int64, xmean::Float64, xsumsq::Float64;
                     a::Float64=1.0, Q::Float64=-1.0)
    # initial guess for optimization: slightly off to avoid
    # ERROR: Linesearch failed to converge
    init = [n * 1.0001, xmean * 1.0005, xsumsq / n * 1.0005]
    # Hessian
    H = Array{Float64}(3,3)

    function integrate(moment::TARGET; Q=-1.0)
        target = targetfactory(moment, n, xmean, xsumsq; a=a, Q=Q)
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
