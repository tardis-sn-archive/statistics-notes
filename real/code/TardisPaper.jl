module TardisPaper

"""reload("TardisPaper"); TardisPaper.GammaIntegrand.test()"""
module GammaIntegrand

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
end

end # GammaIntegrand

end # TardisPaper
