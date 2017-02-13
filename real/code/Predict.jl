"Predictive distribution of Q with various methods"
module Predict

using ..GammaIntegrand, ..Integrate
import DiffBase, Distributions, Optim

using Logging

""" Find initial value of N that will likely give the largest
contribution to (gmhb). It's only approximate but should be a good
starting value

For N>=1, the value is just returned. Else it is inferred by a
heuristic.

"""
function initialize_N(Q, α, β, a, nb, N=0)
    if N < 1
        # perform optimization
        res = GammaIntegrand.heuristicN(Q, α, β, a, nb)
        N = convert(Integer, ceil(Optim.minimizer(res)))
    end

    debug("initial N=$N")
    @assert(N >= 1)
    return N
end

"""Add contribution `f(N)` to `res`.

TODO type of function for type stability?
"""
function search(res, N, f, ε)
    # N>0 required to call f
    N > 0 || return false, 0
    latest = f(N)
    debug("N=$N: now=$latest, res=$res")
    return (latest / res) > ε, latest
end

"Iteratively update `N` from `Ninit` until contribution of `f` is negligible"
function iterate(Ninit, f, ε)
    # initialization
    Nup = Ninit
    Ndown = Ninit
    res = 0.0

    # start at initial N, and update res
    _, res = search(res, Ninit, f, ε)
    res > 0.0 || error("Got exactly zero at initial N $Ninit")

    # then go up and down until contribution negligible
    goup, godown = true, true
    latest = 0.0
    while goup || godown
        if goup
            Nup += 1
            goup, latest = search(res, Nup, f, ε)
            res += latest
        end
        if godown
            Ndown -= 1
            godown, latest = search(res, Ndown, f, ε)
            res += latest
        end
    end
    return res, Ndown, Nup
end

"""
Predict Q by summing over N and considering α, β fixed.
(gmhb)
"""
function by_sum(Q::Real, α::Real, β::Real, a::Real, nb::Real; Ninit::Real=0, ε::Real=1e-3)
    Ninit = initialize_N(Q, α, β, a, nb, Ninit)
    f = N -> exp(log_poisson_predict(N, nb, a) + log_gamma_predict(Q, α, β, N))
    iterate(Ninit, f, ε)
end

"""
Predict Q by summing over N and integrating over α, β  with the Laplace approximation.
(spk)
"""
function by_laplace(Q, a, n, q, logr, nb=n; Ninit=0, ε=1e-3)
    # TODO max. likelihood values for α, β as initial point

    # need evidence for normalized posterior
    res, diffstore = GammaIntegrand.optimize_log_posterior(n, q, logr)
    H = DiffBase.hessian(diffstore)
    Z = Integrate.by_laplace(-Optim.minimum(res), H)

    mode = Optim.minimizer(res)
    invH = inv(H)
    debug("mode ± std. err: α=" , mode[1], "±", invH[1,1], ", β=", mode[2], "±", invH[2,2])

    # find initial N at posterior mode of α, β
    α, β = Optim.minimizer(res)
    Ninit = initialize_N(Q, α, β, a, nb, Ninit)

    # integrate by Laplace on log scale
    f = N -> begin
        kwargs = Dict(:αinit=>α, :βinit=>β)
        res, diffstore = optimize_log_posterior_predict(n, q, logr, Q, N, Z; kwargs...)
        integral = Integrate.by_laplace(-Optim.minimum(res), DiffBase.hessian(diffstore))
        exp(log_poisson_predict(N, nb, a) + integral)
    end
    iterate(Ninit, f, ε)
end

"""
Predict Q by summing over N and integrating over α, β with cubature rules
(spk).

Arguments
---------

`ε`: tolerance for truncating the sum over N
`reltol`: tolerance for cubature integration
"""

function by_cubature(Q, a, n, q, logr, nb=n; αmin=1e-2, αmax=5, βmin=1e-5, βmax=100,
                     Ninit=0, ε=1e-3, reltol=1e-3)
    # to avoid overflow in Cubature, evaluate target at mode and subtract it. The
    # value within a few order of magnitude of
    fit_dist = Distributions.fit_mle(Distributions.Gamma, Distributions.GammaStats(q, logr, n))
    α, β = Distributions.params(fit_dist)
    # from scale to rate parameter
    β = 1/β
    debug("MLE estimates $α, $β")
    logf_mode = log_posterior(α, β, n, q, logr)

    # TODO take limits like 5*std. err. from Gaussian approximation
    # problem: est. uncertainty in β is ok but way too small in α
    # truth: 1.5, 60. With 1000 samples
    # mode ± std. err: α=1.589259406326531±0.004222449140609555, β=65.64769067441536±9.916379800578136
    lower = [αmin, βmin]
    upper = [αmax, βmax]

    Z, σ, ncalls = Integrate.by_cubature(make_log_posterior(n, q, logr, logf_mode), lower, upper; reltol=reltol)
    debug("cubature: $(logf_mode), $Z, $σ, $ncalls")

    # actual evidence larger by logf because we already subtracted it
    # in log_posterior. We need evidence on log scale
    logZ = log(Z) + logf_mode

    # find initial N at posterior mode of α, β
    Ninit = initialize_N(Q, α, β, a, nb, Ninit)

    # integrate by Cubature on linear scale
    f = N -> begin
        integrand = GammaIntegrand.make_log_posterior_predict(n, q, logr, Q, N, logZ)
        integral, σ, ncalls = Integrate.by_cubature(integrand, lower, upper; reltol=reltol)
        debug("integrand: $integral  $σ after $ncalls calls")
        exp(log_poisson_predict(N, nb, a)) * integral
    end
    iterate(Ninit, f, ε)
end

"""
Predict Q by the asymptotic expression (ral) integrating over λ, μ, and σ² with Laplace.

Arguments
---------

`a₀`: hyperprior for inverse Gamma shape
`b₀`: hyperprior for inverse Gamma rate
"""
function asymptotic_by_laplace(Q, a, n, first, second, nb=n; a₀=0, b₀=0)
    # create integrand
    f = make_asymptotic(Q, n, a, first, second, nb; a₀=a₀, b₀=b₀)

    # optimize integrand
    res, diffstore = optimize_integrand_λμσ²(f; λinit=1.05*(nb-a+1), μinit=1.05*first, σ²init=1.05*n/2*(second-first^2))

    # Hessian at mode
    H = DiffBase.hessian(diffstore)
    invH = inv(H)
    mode = Optim.minimizer(res)
    debug(" asympt. Laplace, mode ± std. err: λ=", mode[1], "±", invH[1,1], ", μ=" , mode[2], "±", invH[2,2], ", σ²=", mode[3], "±", invH[3,3])

    # Laplace approximation on log, go back to linear
    exp(Integrate.by_laplace(-Optim.minimum(res), H))
end

function asymptotic_by_cubature(Q, a, n, first, second, nb=n;
                                reltol=1e-3, a₀=0, b₀=0,
                                λmin=0.0, λmax=0.0,
                                μmin=0.0, μmax=0.0,
                                σ²min=0.0, σ²max=0.0)
    # create integrand
    f = make_asymptotic(Q, n, a, first, second, nb; a₀=a₀, b₀=b₀)

    # to avoid overflow in Cubature, evaluate target at approximate mode and subtract it. The
    # value within a few order of magnitude of
    mode = triple_mode(nb, first, second)
    logf_mode = f(mode)
    # logf_mode = 0.0
    debug("asymptotic_by_cubature logf($mode)=$(logf_mode)")
    target = x->f(x) - logf_mode

    lower, upper = GammaIntegrand.triple_ranges(n, first, second, nb)
    (λmin > 0.0) && (lower[1] = λmin)
    (λmax > λmin && λmax > 0.0) && (upper[1] = λmax)
    (μmin > 0.0) && (lower[2] = μmin)
    (μmax > μmin && μmax > 0.0) && (upper[2] = μmax)
    (σ²min > 0.0) && (lower[3] = σ²min)
    (σ²max > σ²min && σ²max > 0.0) && (upper[3] = σ²max)

    Z, σ, ncalls = Integrate.by_cubature(target, lower, upper; reltol=reltol)

    # now we have to undo max. subtraction
    Z *= exp(logf_mode)
    debug("cubature asymptotic: $Z, $σ, $ncalls")
    Z
end

end #Predict
