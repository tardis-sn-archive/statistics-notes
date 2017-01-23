module TardisPaperTest

import TardisPaper.Integrate, TardisPaper.Predict
using TardisPaper.GammaIntegrand
using Base.Test, DiffBase, Distributions, ForwardDiff, Optim # Compat

function integrate_by_cubature()
    res, σ, ncalls = Integrate.by_cubature(x -> log(x^2), 0, 1)
    println(res, " ", σ, " ", ncalls)
    @test_approx_eq_eps(res, 1/3, 1e-15)
end

function predict_laplace()
    α = 1.5
    β = 60.
    n = 500
    a = 1/2
    Q = n*α/β
    ε = 1e-3
    srand(1612)

    dist = Gamma(α, 1/β)
    samples = rand(dist, n)
    q = sum(samples)
    logr = sum(log(samples))

    res, Ndown, Nup = Predict.by_laplace(Q, a, n, q, logr)
end

function predict_alpha_fixed()
    α = 1.5
    β = 60.
    n = 5
    a = 1/2
    Q = n*α/β
    ε = 1e-3

    res, Ndown, Nup = Predict.by_sum(Q, α, β, a, n; ε=ε)
    target = 0.0
    for N in Ndown:Nup
        target += exp(log_poisson_predict(N, n, a) + log_gamma_predict(Q, α, β, N))
    end
    @test res == target

    # adding more terms shouldn't change much
    for N in Nup+1:Nup+10
        target += exp(log_poisson_predict(N, n, a) + log_gamma_predict(Q, α, β, N))
    end
    @test isapprox(res, target; rtol=ε)
end

function gamma_integrand()
    N=7; n=11; a=0;

    @test log_poisson_predict(N, n, a) ≈ log(binomial(N+n-a, N)) - (N+n-a+1)*log(2)

    # draw samples from a Gamma distribution
    α = 1.5
    β = 60
    n = 500
    a = 1/2

    # mode of integrand for N near n at most likely Q
    res = heuristicN(n*α/β, α, β, a, n)
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

    res, diffstore = optimize_log_posterior(length(samples), q, logr)

    # only finite accuracy in max-likelihood
    @test_approx_eq_eps(Optim.minimizer(res)[1], α, 1e-3)
    @test_approx_eq_eps(Optim.minimizer(res)[2], β, 1)

    # but numerical optimizer should find same result as semi-analytic MLE
    dist_fit = Distributions.fit_mle(Gamma, samples)
    @test Optim.minimizer(res)[1] ≈ shape(dist_fit)
    @test Optim.minimizer(res)[2] ≈ 1/scale(dist_fit)

    # compute the evidence
    Z_lap = Integrate.by_laplace(-Optim.minimum(res), DiffBase.hessian(diffstore))

    # cubature needs rescaling as it works on linear scale, and exp(-log(Z))=0 to machine precision
    normalized_f = make_log_posterior(length(samples), q, logr, Z_lap)

    ε = 1e-3
    Z_cub, σ, ncalls = Integrate.by_cubature(normalized_f, [1.2, 50], [2, 70]; reltol=ε)
    println("After $ncalls calls: $(Z_cub) ± $σ")
    # if both agree, cubature should find a normalized posterior
    @test_approx_eq_eps(Z_cub, 1.0, 3ε)

    # integral over prediction should be normalized in 1D...
    target = Q -> log_gamma_predict(Q, α, β, n)
    estimate, σ, ncalls = Integrate.by_cubature(target, 0, Inf; reltol=1e-6)
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

function run()
    @testset "all tests" begin
        integrate_by_cubature()
        predict_laplace()
        predict_alpha_fixed()
        gamma_integrand()
    end
end

end
