"Predictive distribution of Q with various methods"
module Predict

using ..GammaIntegrand, ..Integrate
import DiffBase, Distributions, Optim

using Logging
@Logging.configure(level=DEBUG)

""" Find initial value of N that will likely give the largest
contribution to (gmhb). It's only approximate but should be a good
starting value

For N>=1, the value is just returned. Else it is inferred by a
heuristic.

"""
function initialize_N(Q, α, β, a, n, N=0)
    if N == 0
        # perform optimization
        res = GammaIntegrand.heuristicN(Q, α, β, a, n)
        N = convert(Integer, ceil(Optim.minimizer(res)))
    end

    @assert(N >= 1)
    info("initial N=$N")
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
function by_sum(Q::Real, α::Real, β::Real, a::Real, n::Real; Ninit::Real=0, ε::Real=1e-3)
    Ninit = initialize_N(Q, α, β, a, n, Ninit)
    f = N -> exp(log_poisson_predict(N, n, a) + log_gamma_predict(Q, α, β, N))
    iterate(Ninit, f, ε)
end

"""
Predict Q by summing over N and integrating over α, β  with the Laplace approximation.
(spk)
"""
function by_laplace(Q, a, n, q, logr; Ninit=0, ε=1e-3)
    # TODO max. likelihood values for α, β? ∃ closed-form solution? Could use Distribution.fit_mle
    # d = GammaIntegrand.Distributions.Gamma(1, 1)
    # ss = suffstats(typeof(d), samples)

    # need evidence for normalized posterior
    res, diffstore = GammaIntegrand.optimize_log_posterior(n, q, logr)
    Z = Integrate.by_laplace(-Optim.minimum(res), DiffBase.hessian(diffstore))

    # find initial N at posterior mode of α, β
    α, β = Optim.minimizer(res)
    Ninit = initialize_N(Q, α, β, a, n, Ninit)

    # integrate by Laplace on log scale
    f = N -> begin
        kwargs = Dict(:αinit=>α, :βinit=>β)
        res, diffstore = optimize_log_posterior_predict(n, q, logr, Q, N, Z; kwargs...)
        integral = Integrate.by_laplace(-Optim.minimum(res), DiffBase.hessian(diffstore))
        exp(log_poisson_predict(N, n, a) + integral)
    end
    iterate(Ninit, f, ε)
end

"""
Predict Q by summing over N and integrating over α, β  with cubature rules
(spk)

Arguments
---------

`reltol`: tolerance for cubature integration
"""

function by_cubature(Q, a, n, q, logr; αmin=1.3, αmax=1.8, βmin=40, βmax=100,
                     Ninit=0, ε=1e-3, reltol=1e-3)
    # to avoid overflow in Cubature, evaluate target at mode and subtract it. The
    # value within a few order of magnitude of
    fit_dist = Distributions.fit_mle(Distributions.Gamma, Distributions.GammaStats(q, logr, n))
    α, β = Distributions.params(fit_dist)
    # from scale to rate parameter
    β = 1/β
    debug("MLE estimates $α, $β")
    logf_mode = log_posterior(α, β, n, q, logr)

    lower = [αmin, βmin]
    upper = [αmax, βmax]

    Z, σ, ncalls = Integrate.by_cubature(make_log_posterior(n, q, logr, logf_mode), lower, upper; reltol=reltol)
    debug("cubature: $(logf_mode), $Z, $σ, $ncalls")

    # actual evidence larger by logf because we already subtracted it
    # in log_posterior. We need evidence on log scale
    logZ = log(Z) + logf_mode

    # find initial N at posterior mode of α, β
    Ninit = initialize_N(Q, α, β, a, n, Ninit)

    # integrate by Cubature on linear scale
    f = N -> begin
        # res, diffstore = optimize_log_posterior_predict(n, q, logr, Q, N, Z; kwargs...)
        # integral = Integrate.by_laplace(-Optim.minimum(res), DiffBase.hessian(diffstore))
        integrand = GammaIntegrand.make_log_posterior_predict(n, q, logr, Q, N, logZ)
        integral, σ, ncalls = Integrate.by_cubature(integrand, lower, upper; reltol=reltol)
        debug(integral, σ, ncalls)
        exp(log_poisson_predict(N, n, a)) * integral
    end
    iterate(Ninit, f, ε)
end

end #Predict
