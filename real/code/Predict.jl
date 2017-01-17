module Predict

using ..GammaIntegrand
import Optim
using Base.Test

""" Find initial value of N that will likely give the largest
contribution to (gmhb). It's only approximate but should be a good
starting value

For N>=1, the value is just returned. Else it is inferred by a
heuristic.

"""
function initialize_N(Q, α, β, a, n, N=0)
    if N == 0
        # perform optimization
        res = GammaIntegrand.solve(Q, α, β, a, n)
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
    println("N=$N: now=$latest, res=$res")
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
(gmhb), just sum and consider α, β fixed.
"""
function by_sum(Q::Real, α::Real, β::Real, a::Real, n::Real; Ninit::Real=0, ε::Real=1e-3)
    Ninit = initialize_N(Q, α, β, a, n, Ninit)
    f = N -> exp(log_poisson_predict(N, n, a) + log_gamma_predict(Q, α, β, N))
    iterate(Ninit, f, ε)
end

function test_alpha_fixed()
    α = 1.5
    β = 60.
    n = 5
    a = 1/2
    Q = n*α/β
    ε = 1e-3

    res, Ndown, Nup = by_sum(Q, α, β, a, n; ε=ε)
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

"""
(spk), α, β integrated out with Laplace

TODO find elegant generic programming technique to not rewrite the structure of by_sum. We only need to replace the search function but it is a closure.
"""
function by_laplace(Q::Real, a::Real, n::Real; Ninit::Real=0, ε::Real=1e-3)

end

end #Predict
