module GaussIntegrand

using GammaIntegrand: log_inv_gamma, log_normal, optimize_integrand_αβ

const a0def = 0.0
const b0def = 0.0

"p(μ, σ^2)"
log_prior(μ, σ2, μmax, a0=a0def, b0=b0def) = (a0 == a0def && b0 == b0def) ? 0 : log_inv_gamma(σ2, a0, b0)

"p(μ, σ^2 | k, <ℓ>, V[ℓ])"
log_posterior(μ, σ2, k, meanℓ, varℓ, a0=a0def, b0=b0def) = log_normal(μ, meanℓ, σ2/k) + log_inv_gamma(σ2, a0 + k/2, b0 + k/2*varℓ)

function make_true_posterior(Q0, n, a, k, meanℓ, varℓ, a0=a0def, b0=b0def)
    function (θ::Vector)
        μ, σ2 = θ
        log_posterior(μ, σ2, k, meanℓ, varℓ, a0, b0) + log_gamma(Q0/μ, n-a+1, 1) - log(μ)
    end
end

function optimize_true_posterior(Q0, n, a, k, meanℓ, varℓ, a0=a0def, b0=b0def; kwargs...)
    optimize_integrand_αβ(make_true_posterior(Q0, n, a, k, meanℓ, varℓ, a0=a0def, b0=b0def; kwargs...))
end

end # GaussIntegrand
