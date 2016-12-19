module TardisPaper

module GammaPosterior
"""reload("TardisPaper"); TardisPaper.GammaPosterior.test()"""

export log_posterior, make_log_posterior

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

    @test isapprox(Optim.minimizer(res)[1], α; rtol=1e-3)
    @test isapprox(Optim.minimizer(res)[2], β; rtol=5e-2)
end

end # GammaPosterior
end # TardisPaper
