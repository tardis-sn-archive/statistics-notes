module Predict

using ..GammaIntegrand
import Optim
using Base.Test

"""
(gmhb), just sum and consider α, β fixed.
"""
function by_sum(Q::Real, α::Real, β::Real, a::Real, n::Real; Ninit::Real=0, ε::Real=1e-3, verbose=false)
    if Ninit == 0
        # perform optimization, optim only does minimization but we
        # actually optimize to find the N that will likely give the
        # largest contribution. It's only approximate but should be a
        # good starting value
        res = GammaIntegrand.solve(Q, α, β, a, n)
        # println(res)
        # println((Q, α, β, a, n))
        Ninit = convert(Integer, ceil(Optim.minimizer(res)))
    end

    @assert(Ninit >= 1)
    info("initial N=$Ninit")

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
        verbose && println("N=$N: now=$latest, res=$res")

        return (latest / res) > ε
    end

    # start at initial N, and update res
    _ = search(Ninit)
    res > 0 || error("Got exactly zero at initial N $Ninit")
    # then go up and down until contribution negligible
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

function test_alpha_fixed()
    α = 1.5
    β = 60.
    n = 5
    a = 1/2
    Q = n*α/β
    ε = 1e-3

    res, Ndown, Nup = by_sum(Q, α, β, a, n; ε=ε, verbose=true)
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
function by_laplace(Q::Real, a::Real, n::Real; Ninit::Real=0, ε::Real=1e-3, verbose=false)

end

end #Predict
