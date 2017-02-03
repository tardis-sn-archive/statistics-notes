"""reload("TardisPaper"); TardisPaper.GammaIntegrand.test()"""
module GammaIntegrand

export heuristicN, log_gamma, log_gamma_predict, log_inv_gamma, log_normal, log_poisson_predict, log_posterior
export make_asymptotic, make_log_posterior
export optimize_log_posterior, optimize_log_posterior_predict, optimize_integrand_λμσ², triple_ranges

import ..Integrate

using DiffBase, ForwardDiff, Optim

""" posterior p(α, β | n, q, r) for the parameters of the Gamma
distribution given the sufficient statistics of the samples."""
log_posterior(α::Real, β::Real, n::Real, q::Real, logr::Real, evidence=0.0) = n*(α*log(β)-lgamma(α)) + (α-1)*logr - β*q - evidence
function make_log_posterior(n::Real, q::Real, logr::Real, evidence=0.0)
    function (θ::Vector)
        α, β = θ
        res = log_posterior(α, β, n, q, logr, evidence)
        # println("got $res for $α $β")
        isnan(res) && error("Got NaN for $a $b")
        res
    end
end

""" Gamma(x | α, β) """
log_gamma(x::Real, α::Real, β::Real) = α*log(β) - lgamma(α) + (α-1)*log(x) - β*x

""" p(Q | α, β, N) """
log_gamma_predict(Q::Real, α::Real, β::Real, N::Real) = log_gamma(Q, N*α, β)

"""(pmh) p(N | n)"""
function log_poisson_predict(N::Real, n::Integer, a::Real)
    tmp = N+n-a+1
    lgamma(tmp) - lgamma(N+1) - lgamma(n-a+1) - tmp * log(2)
end

"Return Optimize.result struct"
function optimize_integrand(target, lower, upper, initial)
    min_target = x -> -target(x)
    # TODO couldn't get autodiff to work here. Probably because of the box constraints
    res = optimize(DifferentiableFunction(min_target), initial,
                   lower, upper, Fminbox(), optimizer=LBFGS)
                   # optimizer_o=OptimizationOptions(autodiff=true))
    # get gradient and Hessian into one result
    storage = DiffBase.HessianResult(initial)
    ForwardDiff.hessian!(storage, min_target, Optim.minimizer(res))
    res, storage
end

function optimize_integrand_αβ(target; αmin=0.0, αmax=Inf, βmin=0.0, βmax=Inf, αinit=1.5, βinit=50.0)
    lower = [αmin, βmin]
    upper = [αmax, βmax]
    initial = [αinit, βinit]
    optimize_integrand(target, lower, upper, initial)
end

function optimize_integrand_λμσ²(target;
                                λmin=0.0, λmax=Inf, λinit=50.0,
                                μmin=0.0, μmax=Inf, μinit=1.5/50,
                                σ²min=1e-6, σ²max=Inf, σ²init=1.5/50^2)
    lower = [λmin, μmin, σ²min]
    upper = [λmax, μmax, σ²max]
    initial = [λinit, μinit, σ²init]
    optimize_integrand(target, lower, upper, initial)
end

function optimize_log_posterior(n::Real, q::Real, logr::Real; kwargs...)
    optimize_integrand_αβ(make_log_posterior(n, q, logr); kwargs...)
end

function make_log_posterior_predict(n, q, logr, Q, N, evidence=0.0)
    function (θ::Vector)
        α, β = θ
        log_posterior(α, β, n, q, logr, evidence) + log_gamma_predict(Q, α, β, N)
    end
end

function optimize_log_posterior_predict(n, q, logr, Q, N, evidence=0.0; kwargs...)
    optimize_integrand_αβ(make_log_posterior_predict(n, q, logr, Q, N, evidence); kwargs...)
end

"""
Collect data that the target density needs
"""
# TODO not type stable, use parametric type
type OptData
    Q::Real
    α::Real
    β::Real
    a::Real
    n::Integer
    count::Integer

    function OptData(Q::Real, α::Real, β::Real, a::Real, n::Integer)
        if Q < 0 Error("Invalid Q < 0: $Q") end
        if α <= 0 Error("Invalid α <= 0: $α") end
        if β <= 0 Error("Invalid β <= 0: $β") end
        if a < 0 Error("Invalid a < 0: $a") end
        new(Q, α, β, a, n, 0)
    end
end

"""
Create the target for optimization of the integrand of (spk) as a function of N. Negate for minimization
"""
function factory(Q::Real, α::Real, β::Real, a::Real, n::Integer)
    popt = OptData(Q, α, β, a, n)
    f(N::Real) = (popt.count += 1; -log_poisson_predict(N, popt.n, popt.a) - log_gamma_predict(popt.Q, popt.α, popt.β, N))
    return popt, f
end

"""
Find the arg max_N NegativeBinomial(N|n-a+1, 1/2)*Gamma(Q|Nα, β) using Brent's method.

Use the plug-in values of α and β to avoid costly integration.
"""
function heuristicN(Q::Real, α::Real, β::Real, a::Real, n::Integer;
                    ε=1e-2, min=0.0, max=0.0, trace=false)
    popt, f = factory(Q, α, β, a, n)
    if (max <= 0.0) max = 10n end
    optimize(f, min, max, rel_tol=ε, show_trace=trace, extended_trace=trace)
end

log_normal(x, μ, σ²) = σ²<=0 ? -Inf : -0.5(log(2π*σ²) + (x-μ)^2/σ²)

"Log pdf of InverseGamma distribution with shape parameter `a` and scale parameter `b`."
log_inv_gamma(x, a, b) = x <= 0? -Inf : a*log(b) - lgamma(a) - (a+1)*log(x) - b/x

"""
Integrand of (ral), negated for minimization.

# Arguments

* `first`: first sample moment
* `second`: first sample moment
* `a₀`: hyperprior for inverse Gamma shape
* `b₀`: hyperprior for inverse Gamma rate
"""
function make_asymptotic(Q, n, a, first, second; a₀=0, b₀=0)
    # (rbd)
    μₙ = first
    aₙ = a₀+n/2
    bₙ = b₀+n/2*(second-first^2)
    function(x)
        λ, μ, σ² = x
        # (rbd)
        σₙ² = σ²/n
        # (ral)
        log_normal(Q, λ*μ, λ*(σ²+μ^2)) + log_gamma(λ,n-a+1,1) + log_normal(μ, μₙ, σₙ²) + log_inv_gamma(σ², aₙ, bₙ)
    end
    end

    "Extract decent ranges on [λ, μ, σ²] from marginal posterior distributions. "
    function triple_ranges(n, first, second; k=5)
        mode = [n, first, second-first^2]

        lower = zeros(3)
        upper = zeros(lower)

        # get ranges

        # Poisson λ
        Δ = k*sqrt(n)
        lower[1] = max(n - Δ, 0.0)
        upper[1] = n + Δ

        # Gaussian μ
        Δ = k*sqrt(mode[3]/n)
        lower[2] = max(mode[2] - Δ, 0.0)
        upper[2] = mode[2] + Δ

        # InverseGamma σ²
        Δ = k*sqrt((n/2 * mode[3])^2 / ((n/2 - 1)^2 * (n/2 - 2)))
        lower[3] =  max(mode[3] - Δ, 0.0)
        upper[3] = mode[3] + Δ

        lower, upper
    end
end # GammaIntegrand
