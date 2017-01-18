"""reload("TardisPaper"); TardisPaper.GammaIntegrand.test()"""
module GammaIntegrand

export heuristicN, log_gamma, log_gamma_predict, log_poisson_predict, log_posterior, make_log_posterior
export optimize_log_posterior, optimize_log_posterior_predict

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
function optimize_integrand(target; αmin=1.0, αmax=Inf, βmin=0, βmax=Inf, αinit=1.5, βinit=50)
    min_target = x -> -target(x)

    lower = [αmin, βmin]
    upper = [αmax, βmax]
    initial_θ = [αinit, βinit]
    # TODO couldn't get autodiff to work here. Probably because of the box constraints
    res = optimize(DifferentiableFunction(min_target), initial_θ,
                   lower, upper, Fminbox(), optimizer=LBFGS)
                   # optimizer_o=OptimizationOptions(autodiff=true))
    # get gradient and Hessian into one result
    storage = DiffBase.HessianResult(initial_θ)
    ForwardDiff.hessian!(storage, min_target, Optim.minimizer(res))
    res, storage
end

function optimize_log_posterior(n::Real, q::Real, logr::Real; kwargs...)
    optimize_integrand(make_log_posterior(n, q, logr); kwargs...)
end

function make_log_posterior_predict(n, q, logr, Q, N, evidence=0.0)
    function (θ::Vector)
        α, β = θ
        log_posterior(α, β, n, q, logr, evidence) + log_gamma_predict(Q, α, β, N)
    end
end

function optimize_log_posterior_predict(n, q, logr, Q, N, evidence=0.0; kwargs...)
    optimize_integrand(make_log_posterior_predict(n, q, logr, Q, N, evidence); kwargs...)
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
Find the arg max_N NegativeBinomial(N|n-a+1, 1/2)*Gamma(Q|Nα, β) using Brent's method.

Use the plug-in values of α and β to avoid costly integration.
"""
function heuristicN(Q::Real, α::Real, β::Real, a::Real, n::Integer;
                    ε=1e-2, min=0.0, max=0.0, trace=false)
    popt, f = factory(Q, α, β, a, n)
    if (max <= 0.0) max = 10n end
    optimize(f, min, max, rel_tol=ε, show_trace=trace, extended_trace=trace)
end

end # GammaIntegrand
