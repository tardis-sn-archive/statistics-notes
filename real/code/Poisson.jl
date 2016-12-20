module Poisson

using Base.Test

"""(pmh)"""
function log_predictive(N::Real, n::Integer, a::Real)
    tmp = N + n -a + 1
    lgamma(tmp) - lgamma(N+1) - lgamma(n-a+1) - tmp * log(2)
end

function test()
    N=7; n=11; a=0;

    @test log_predictive(N, n, a) ≈ log(binomial(N+n-a, N)) - (N+n-a+1)*log(2)
end

end #Poisson
