"""reload("TardisPaper"); TardisPaper.GammaIntegrand.test()"""
module GammaIntegrand

export log_gamma, log_gamma_predict, log_poisson_predict, log_posterior, make_log_posterior

import ..Integrate
using Base.Test, Distributions, Optim

""" posterior p(α, β | n, q, r) for the parameters of the Gamma
distribution given the sufficient statistics of the samples."""
log_posterior(α::Real, β::Real, n::Real, q::Real, logr::Real) = n*(α*log(β)-lgamma(α)) + (α-1)*logr - β*q
function make_log_posterior(n::Real, q::Real, logr::Real)
    function (θ::Vector)
        α::Real, β::Real = θ
        log_posterior(α, β, n, q, logr)
    end
end

"Return Optimize.result struct"
function optimize_log_posterior(n::Real, q::Real, logr::Real;
                                αmin=1.0, αmax=Inf, βmin=0, βmax=Inf,
                                αinit=1.5, βinit=50)
    opt_target = make_log_posterior(n, q, logr)
    min_target = x -> -opt_target(x)

    lower = [αmin, βmin]
    upper = [αmax, βmax]
    initial_θ = [αinit, βinit]
    optimize(DifferentiableFunction(min_target), initial_θ,
             lower, upper, Fminbox(), optimizer=LBFGS)
end

""" Gamma(x | α, β) """
log_gamma(x::Real, α::Real, β::Real) = α*log(β) - lgamma(α) + (α-1)*log(x) - β*x

""" p(Q | α, β, N) """
log_gamma_predict(Q::Real, α::Real, β::Real, N::Real) = log_gamma(Q, N*α, β)

"""(pmh) p(N | n)"""
function log_poisson_predict(N::Real, n::Integer, a::Real)
    tmp = N + n -a + 1
    lgamma(tmp) - lgamma(N+1) - lgamma(n-a+1) - tmp * log(2)
end

"""
Collect data that the target density needs
"""
type OptData
    Q::Real
    α::Real
    β::Real
    a::Real
    n::Integer
    count::Integer

    function OptData(Q::Real, α::Real, β::Real, a::Real, n::Integer)
        if Q < 0 Error("Invalid Q < 0: $Q") end
        if α < 1 Error("Invalid α < 1: $α") end
        if β < 0 Error("Invalid β < 0: $β") end
        if a < 0 Error("Invalid a < 0: $a") end
        new(Q, α, β, a, n, 0)
    end
end

"""
Create the target with data embedded via a closure. Negate for minimization
"""
function factory(Q::Real, α::Real, β::Real, a::Real, n::Integer)
    popt = OptData(Q, α, β, a, n)
    f(N::Real) = (popt.count += 1; -log_poisson_predict(N, popt.n, popt.a) - log_gamma_predict(popt.Q, popt.α, popt.β, N))
    return popt, f
end

"""
Find the arg max_N NegativeBinomial(N|n-a+1, 1/2)*Gamma(X|Nα, β) using Brent's method.

Use the plug-in values of α and β to avoid costly integration.
"""
function solve(Q::Real, α::Real, β::Real, a::Real, n::Integer;
               ε=1e-2, min=0.0, max=0.0, trace=false)
    popt, f = factory(Q, α, β, a, n)
    if (max <= 0.0) max = 10n end
    optimize(f, min, max, rel_tol=ε, show_trace=trace, extended_trace=trace)
end

function test()
    N=7; n=11; a=0;

    @test log_poisson_predict(N, n, a) ≈ log(binomial(N+n-a, N)) - (N+n-a+1)*log(2)

    # draw samples from a Gamma distribution
    α = 1.5
    β = 60
    n = 500
    a = 1/2

    # mode of integrand for N near n at most likely Q
    res = solve(n*α/β, α, β, a, n)
    # println(res)
    @test isapprox(Optim.minimizer(res), n; rtol=1e-3)

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
    @test log_gamma_predict(x, α, β, n) ≈ logpdf(Gamma(n*α, 1/β), x)

    @test log_posterior(α, β, n, q, logr) ≈ 1407.87731905469
    # TODO was type stable before, why not anymore?
    # @inferred log_posterior(α, β, n, q, logr)

    f = make_log_posterior(n, q, logr)
    @test f([α, β]) == log_posterior(α, β, n, q, logr)

    # box minimization with more samples
    samples = rand(dist, n*50)
    q = sum(samples)
    logr = sum(log(samples))

    res = optimize_log_posterior(length(samples), q, logr)
    # println(res)

    # only finite accuracy in max-likelihood
    @test isapprox(Optim.minimizer(res)[1], α; rtol=1e-3)
    @test isapprox(Optim.minimizer(res)[2], β; rtol=5e-2)

    # integral over prediction should be normalized in 1D...
    estimate, σ, ncalls = Integrate.by_cubature(Q -> log_gamma_predict(Q, α, β, n), 0, Inf; reltol=1e-6)
    # println("ncalls $ncalls")
    @test isapprox(estimate, 1; atol=σ)

    # and 2D for large but finite upper limits ...
    estimate, σ, ncalls = Integrate.by_cubature(x -> log_gamma_predict(x[1], α, β, n) + log_gamma_predict(x[2], α, β, n),
                                                [0.0, 0.0], [500.0, 500.0]; reltol=1e-6)
    println("After $ncalls calls: $estimate ± $σ")
    @test isapprox(estimate, 1; atol=σ)

    # and infinite upper limits.
    estimate, σ, ncalls = Integrate.by_cubature(x -> log_gamma_predict(x[1], α, β, n) + log_gamma_predict(x[2], α, β, n),
                                                [0.0, 0.0], [Inf, Inf]; reltol=1e-6)
    println("After $ncalls calls: $estimate ± $σ")
    @test isapprox(estimate, 1; atol=σ)
end

end # GammaIntegrand