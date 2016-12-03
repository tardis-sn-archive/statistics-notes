module Asymptotic

using tardis, Optim
import ConjugatePriors
import ForwardDiff
chunk = ForwardDiff.Chunk{3}()


"""
Enumeration to specify which target function to create
"""
@enum TARGET Posterior FirstMoment SecondMoment ThirdMoment FourthMoment

"""
Create a function for Laplace approximation
"""
function targetfactory(target::TARGET, n::Int64, xmean::Float64, xsumsq::Float64, a::Float64=1)
    0 <= a <= 1 || error("need a in [0,1]")
    (xmean > 0) || error("nonpositive xmean $xmean")
    (xsumsq > 0) || error("nonpositive xsumsq $xsumsq")

    """ with X (or Q) dependence, only contains λ, μ, σSq posterior"""
    function logposterior(x::Vector)
        λ, μ, σSq = x

        # gamma(λ), set β=1
        α = n-a+1
        resgamma = - lgamma(α) + (α-1)*log(λ) - λ
        ## println(resgamma)

        # normal-inversegamma
        d = ConjugatePriors.NormalInverseGamma(xmean, n, n/2, xsumsq/2)

        resnormal = -1/2 * (log(2*π*σSq / d.v0) + (μ-d.mu)^2 * d.v0 / σSq)
        resinvgamma = -lgamma(d.shape) + d.shape * log(d.scale) - (d.shape + 1) * log(σSq) - d.scale / σSq

        ## println(resinvgamma," ", resnormal)
        return resgamma + resnormal + resinvgamma
    end

    """log(-d/dt f(target=0|x))"""
    function logfirst(x::Vector)
        λ, μ, σSq = x
        # bug fix
        σSq += μ^2
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) * sqrt(λ * σSq / (2 * π)) + 1/2 * λ * μ * (1 + erf(sqrt(exponent)))
        log(tmp)
    end

    """log(d²/dt² f(target=0|x))"""
    function logsecond(x::Vector)
        λ, μ, σSq = x
        # bug fix
        σSq += μ^2
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) * sqrt(2 / π * λ^3 * σSq) * μ + λ*(λ*μ^2 + σSq)*(1 + erf(sqrt(exponent)))
        log(1/2 * tmp)
    end

    """log(-d^3/dt^3 f(target=0|x))"""
    function logthird(x::Vector)
        λ, μ, σSq = x
        # bug fix
        σSq += μ^2
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) / sqrt(2π) * λ^(3/2) * sqrt(σSq) * (λ*μ^2 + 2σSq)
        tmp += 1/2 * λ * μ * (λ*μ^2 + 3σSq)*(1 + erf(sqrt(exponent)))
        log(tmp)
    end

    """log(d^4/dt^4 f(target=0|x))"""
    function logfourth(x::Vector)
        λ, μ, σSq = x
        # bug fix
        σSq += μ^2
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) / sqrt(2π) * λ^(5/2) * μ * sqrt(σSq) * (λ*μ^2 + 5σSq)
        tmp += 1/2 * λ^2 * (λ^2*μ^4 + 6λ*μ^2 + 3σSq)*(1 + erf(sqrt(exponent)))
        log(tmp)
    end

    # optimizer minimizes => add minus sign at the last possible moment
    if target === Posterior
        return x -> -logposterior(x)
    elseif target === FirstMoment
        return x -> -(logposterior(x) + logfirst(x))
    elseif target === SecondMoment
        return x -> -(logposterior(x) + logsecond(x))
    elseif target === ThirdMoment
        return x -> -(logposterior(x) + logthird(x))
    elseif target === FourthMoment
        return x -> -(logposterior(x) + logfourth(x))
    end
end

# TODO safe Hessian in buffer passed by user
function uncertainty(n::Int64, xmean::Float64, xsumsq::Float64, a::Float64=1.0)
    # initial guess for optimization: slightly off to avoid
    # ERROR: Linesearch failed to converge
    init = [n * 1.0001, xmean * 1.0005, xsumsq / n * 1.0005]
    # Hessian
    H = Array{Float64}(3,3)

    function integrate(moment::TARGET)
        target = targetfactory(moment, n, xmean, xsumsq, a)
        res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
        ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
        exp(tardis.laplace(-Optim.minimum(res), H))
    end

    # mean
    μ = integrate(FirstMoment)

    # second moment
    # target = targetfactory(SecondMoment, n, xmean, xsumsq, a)
    # res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
    # ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
    # logsecond = tardis.laplace(-Optim.minimum(res), H)
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
