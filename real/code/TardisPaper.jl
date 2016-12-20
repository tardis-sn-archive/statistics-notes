module TardisPaper

"""use Cubature or Laplace"""
module Integrate
using Cubature

function by_cubature(logf::Function, xmin::Real, xmax::Real; kwargs...)
    # count function evaluations
    counter = 0
    h = t -> (counter += 1; exp(logf(t)))

    # transform variables
    if xmax === Inf
        h = t -> (counter += 1; exp(logf(xmin + t/(1-t))) * 1/(1-t)^2)
        xmin, xmax = 0, 1
    end
    estimate, σ = hquadrature(h, xmin, xmax; kwargs...)
    return estimate, σ, counter
end

function by_cubature(logf::Function, xmin::Vector, xmax::Vector; kwargs...)
    # count function evaluations
    counter = 0

    # keep independent limits in case of transform
    a = copy(xmin)
    b = copy(xmax)

    # count parameters to transform and their index in the arg vector
    ninf = sum(x -> x == Inf, xmax)
    indices = Vector{Int16}(ninf)
    offset = 1
    for (i, xi) in enumerate(xmax)
        if xi == Inf
            indices[offset] = i
            offset += 1
            a[i] = 0
            b[i] = 1
        end
    end

    if ninf == 0
        h = function(t) counter += 1; exp(logf(t)) end
    else
        # transform variables, mutate x in place
        x = Vector(xmax)
        trafo = function(t)
            x .= t
            jacobian = 1.0
            for i in indices
                x[i] = xmin[i] + t[i]/(1-t[i])
                jacobian *= 1/(1-t[i])^2
            end
            return jacobian
        end
        h = function(t)
            counter += 1
            jacobian = trafo(t)
            exp(logf(x)) * jacobian
        end
    end
    estimate, σ = hcubature(h, a, b; kwargs...)
    return estimate, σ, counter
end

end # Integrate


"""reload("TardisPaper"); TardisPaper.GammaIntegrand.test()"""
module GammaIntegrand

export log_posterior, make_log_posterior

import ..Integrate
using Base.Test, Distributions, Optim

""" posterior P(α, β | n, q, r) for the parameters of the Gamma
distribution given the sufficient statistics of the samples."""
log_posterior(α::Real, β::Real, n::Real, q::Real, logr::Real) = n*(α*log(β)-lgamma(α)) + (α-1)*logr - β*q
function make_log_posterior(n::Real, q::Real, logr::Real)
    function (θ::Vector)
        α::Real, β::Real = θ
        log_posterior(α, β, n, q, logr)
    end
end

""" Gamma(x | α, β) """
log_gamma(x::Real, α::Real, β::Real) = α*log(β) - lgamma(α) + (α-1)*log(x) - β*x

""" p(Q | α, β, N) """
log_predict(Q::Real, α::Real, β::Real, N::Real) = log_gamma(Q, N*α, β)

function test()
    # draw samples from a Gamma distribution
    α = 1.5
    β = 60
    n = 500
    srand(1612)
    dist = Gamma(α, 1/β)
    samples = rand(dist, n)
    q = sum(samples)
    logr = sum(log(samples))

    # compare to Distributions
    x = 0.04
    dist = Gamma(α, 1/β)
    @test log_gamma(x, α, β) ≈ log(pdf(dist, x))
    @test log_gamma(x, α, β) ≈ logpdf(dist, x)
    @test log_predict(x, α, β, n) ≈ logpdf(Gamma(n*α, 1/β), x)

    @test log_posterior(α, β, n, q, logr) ≈ 1407.87731905469
    @inferred log_posterior(α, β, n, q, logr)

    f = make_log_posterior(n, q, logr)
    @test f([α, β]) == log_posterior(α, β, n, q, logr)

    # box minimization with more samples
    samples = rand(dist, n*50)
    q = sum(samples)
    logr = sum(log(samples))

    f = make_log_posterior(length(samples), q, logr)
    negf(θ) = -f(θ)
    lower = [1.0, 0]
    upper = [Inf, Inf]
    initial_θ = 1.05 * [α, β]
    res = optimize(DifferentiableFunction(negf), initial_θ, lower, upper, Fminbox(), optimizer = LBFGS)

    # only finite accuracy in max-likelihood
    @test isapprox(Optim.minimizer(res)[1], α; rtol=1e-3)
    @test isapprox(Optim.minimizer(res)[2], β; rtol=5e-2)

    # integral over prediction should be normalized in 1D...

    estimate, σ, ncalls = Integrate.by_cubature(Q -> log_predict(Q, α, β, n), 0, Inf; reltol=1e-6)
    println("ncalls $ncalls")
    @test isapprox(estimate, 1; atol=σ)

    # and 2D for large but finite upper limits ...
    estimate, σ, ncalls = Integrate.by_cubature(x -> log_predict(x[1], α, β, n) + log_predict(x[2], α, β, n),
                                                [0.0, 0.0], [500.0, 500.0]; reltol=1e-6)
    println("After $ncalls calls: $estimate ± $σ")
    @test isapprox(estimate, 1; atol=σ)

    # and infinite upper limits.
    estimate, σ, ncalls = Integrate.by_cubature(x -> log_predict(x[1], α, β, n) + log_predict(x[2], α, β, n),
                                                [0.0, 0.0], [Inf, Inf]; reltol=1e-6)
    println("After $ncalls calls: $estimate ± $σ")
    @test isapprox(estimate, 1; atol=σ)
end

end # GammaIntegrand

end # TardisPaper
