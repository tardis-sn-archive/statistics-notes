module TardisPaperTest

import TardisPaper.Integrate, TardisPaper.Predict, TardisPaper.Moments
using TardisPaper.GammaIntegrand
using Base.Test, DiffBase, Distributions, ForwardDiff, Logging, Optim

@Logging.configure(level=INFO)

function integrate_by_cubature()
    res, σ, ncalls = Integrate.by_cubature(x -> log(x^2), 0, 1)
    println(res, " ", σ, " ", ncalls)
    @test_approx_eq_eps(res, 1/3, 1e-15)
end

function predict()
    α = 1.5
    β = 60.
    n = 400
    a = 1/2
    Q = n*α/β
    # ε = 1e-3
    srand(11289)

    dist = Gamma(α, 1/β)
    samples = rand(dist, n)
    q = sum(samples)
    logr = sum(log(samples))

    hello(res, alg) = println("$alg: P($Q|$n, $a, x) = $(res[1]) for N=$(res[2])...$(res[3])")

    res_cuba = Predict.by_cubature(Q, 0.5, n, q, logr; reltol=1e-5)
    hello(res_cuba, "cubature")
    res_laplace = Predict.by_laplace(Q, a, n, q, logr)
    hello(res_laplace, "Laplace")
    # min. and max. N should agree within 1
    for i in 2:3
        @test_approx_eq_eps(res_cuba[i], res_laplace[i], 1)
    end

    # P(Q|...)
    @test_approx_eq_eps(res_cuba[1], res_laplace[1], 3e-2)

    first = q/n
    second = mapreduce(x->x^2, +, samples)/n
    res_asym_laplace = Predict.asymptotic_by_laplace(Q, a, n, first, second)
    hello((res_asym_laplace, 0, 0), "asympt. Laplace")
    @test_approx_eq_eps(res_asym_laplace, res_laplace[1], 3e-2)

    res_asym_cubature = Predict.asymptotic_by_cubature(Q, a, n, first, second; reltol=1e-3)
    hello((res_asym_cubature, 0, 0), "asympt. cubature")
    @test_approx_eq_eps(res_asym_cubature, res_laplace[1], 3e-2)
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

function asymptotic()
    μ = 1.3
    σ = 1.44
    x = 2.011
    normal = Distributions.Normal(μ, σ)

    @test_approx_eq(log_normal(x, μ, σ^2), logpdf(normal, x))

    a = 1.5
    b = 0.013
    invgamma = Distributions.InverseGamma(a, b)
    @test_approx_eq(log_inv_gamma(x, a, b), logpdf(invgamma, x))

    λ = 24.4
    Q = λ*μ
    n = 15
    a = 0.5

    first = 1.02*μ
    second = 1.01*σ^2
    f = make_asymptotic(Q, n, a, first, second)
    normalQ = Distributions.Normal(λ*μ, sqrt(λ*(σ^2 + μ^2)))
    normalμ = Distributions.Normal(first, sqrt(σ^2/n))
    # second arg: rate vs. scale. Irrelevant if =1
    gamma = Distributions.Gamma(n-a+1, 1)
    invgamma = Distributions.InverseGamma(n/2, n/2 * (second - first^2))

    @test_approx_eq(+f([λ, μ, σ^2]), logpdf(normalQ, Q) + logpdf(gamma, λ) + logpdf(normalμ, μ) + logpdf(invgamma, σ^2))

    Predict.asymptotic_by_laplace(Q, a, n, first, second)
end

function moments()
    α = 1.5
    β = 60.
    first = α/β
    # E[x²] = V[x]+E[x]²
    second = α/β^2 * (1+α)
    for n = [80, 200]
        info("Checking n=$n")
        init = [n*1.1, first*1.2, second*1.05]

        # neglect uncertainty on λ, μ, σSq
        # then we should reproduce known result for compound Poisson
        target = x -> exp(-Moments.targetfactory(Moments.FirstMomentRaw, n, first, second)(x))
        @test_approx_eq target([n, first, α/β^2]) n*first
        target = x -> exp(-Moments.targetfactory(Moments.SecondMomentRaw, n, first, second)(x))
        # E[x²] = V[x] + E[x]²
        @test_approx_eq(target([n, first, α/β^2]), n*second + (n*first)^2)

        # now integrate over parameters
        res = optimize(Moments.targetfactory(Moments.PosteriorParam, n, first, second),
                       init, Newton(), Optim.Options(autodiff=true))

        mode = Optim.minimizer(res)
        # mode of inverse Gamma not exactly at variance of true Gamma
        # and mode of posterior not exactly at marginal mode
        a = n/2; b = n/2*(second - first^2)
        @test_approx_eq mode[1:2] [n-1/2, first]
        @test_approx_eq_eps(mode[3], b/(a+1), 1e-3)

        moments =  Moments.uncertainty(n, first, second)
        info("Moments output ", moments)
        @test_approx_eq_eps moments[1] n*first 0.1
        info(n*first)

        # don't understand why but σ is only 2/3 of what
        # Simpson's rule gives for mapping out (spk) and (ral)
        scale = 1.5

        if n == 80
            @test_approx_eq_eps moments[1] 2.01 1e-1
            @test_approx_eq_eps moments[2] 0.41 / scale 2e-2
        end
        if n == 200
            @test_approx_eq_eps moments[1] 5.01 1e-1
            @test_approx_eq_eps moments[2] 0.648 / scale 3e-2
        end
    end
end

function run()
    @testset "all tests" begin
        integrate_by_cubature()
        predict()
        predict_alpha_fixed()
        gamma_integrand()
        asymptotic()
        moments()
    end
end

end
