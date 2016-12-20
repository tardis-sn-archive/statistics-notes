module Predict

using ..GammaIntegrand
import Optim

# """

# Compute contribution to `res` for `N`. Return updated `res` and bool
# if search should continue in this direction.

# """
# function search_step!(frame::DataFrame, Pmean, ∇Pmean!, HPmean!, Hres::Matrix, pm::PosteriorMean,
#                       αOrder::Integer, βOrder::Integer, nb::NegBinom, res::Real, θ::Vector, ε::Real;
#                       xtol_rel=1e-5, optimize=true)
# function search_step!(N::Real, Q::Real, α::Real, β::Real, a::Real, n::Real, ε::Real)
#     # N>0 required to search
#     if N <= 0
#         return res, false
#     end

#     # have to leave the log scale now
#     latest = exp(log_predictive(N, n, a) + log_predict(Q, α, β))
#     res += latest
#     # println("$(pm): latest=$latest, res=$res")

#     return res, (latest / res) > ε
# end

"""
(gmhb), just sum and consider α, β fixed.
"""
function by_sum(Q::Real, α::Real, β::Real, a::Real, n::Real; Ninit::Real=0, ε::Real=1e-3)
    if Ninit == 0
        # perform optimization, optim only does minimization but we
        # actually optimize to find the N that will likely give the
        # largest contribution. It's only approximate but should be a
        # good starting value
        res = GammaIntegrand.solve(Q, α, β, a, n)
        Ninit = convert(Integer, ceil(Optim.minimizer(res)))
    end

    @assert(Ninit >= 1)
    println("initial N=$Ninit")

    # initialization
    Nup = Ninit
    Ndown = Ninit
    res = 0.0

    """Add contribution at `N` to `res`."""
    function search(N::Real)
        # N>0 required to search
        if N <= 0
            return false
        end

        # have to leave the log scale now
        latest = exp(log_poisson_predict(N, n, a) + log_gamma_predict(Q, α, β, N))
        res += latest
        # println("$(pm): latest=$latest, res=$res")

        return (latest / res) > ε
    end

    # start at initial N, then go up and down until contribution negligible
    _ = search(Ninit)
    print("$res from Ninit=$Ninit")
    goup, godown = true, true
    while goup || godown
        if goup
            Nup += 1
            goup = search(Nup)
        end
        if godown
            Ndown -= 1
            godown = search(Ndown)
        end
    end
    return res, Ndown, Nup
end

function test()
    α = 1.5
    β = 60.
    n = 500
    a = 1/2

    res, Ndown, Nup = by_sum(n*α/β, α, β, a, n)
end

end #Predict
