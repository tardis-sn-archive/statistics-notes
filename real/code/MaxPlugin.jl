"""
Find the mode of NegativeBinomial(N|n-a+1, 1/2)*Gamma(X|Nα, β) using Brent's method.

Use the plug-in values of α and β to avoid costly integration.
"""
module MaxPlugin
using ..Tardis, ..Optim
using Distributions

"""
Collect data that the target density needs
"""
type OptData
    X::Real
    α::Real
    β::Real
    a::Real
    n::Integer
    count::Integer

    function OptData(X::Real, α::Real, β::Real, a::Real, n::Integer)
        if X < 0 Error("Invalid X < 0: $X") end
        if α < 1 Error("Invalid α < 1: $α") end
        if β < 0 Error("Invalid β < 0: $β") end
        if a < 0 Error("Invalid a < 0: $a") end
        new(X, α, β, a, n, 0)
    end
end

"""
Create the target with data embedded via a closure
"""
function factory(X::Real, α::Real, β::Real, a::Real, n::Integer)
    popt = OptData(X, α, β, a, n)
    f(N::Real) = (popt.count += 1; -tardis.lognegativebinomial(N, popt.n, popt.a) - loggamma(popt.X, N * popt.α, popt.β))
    return popt, f
end

function solve(X::Real, α::Real, β::Real, a::Real, n::Integer;
               ε=1e-2, min=0.0, max=0.0, trace=false)
    popt, f = factory(X, α, β, a, n)
    if (max <= 0.0) max = 10n end
    optimize(f, min, max, rel_tol=ε, show_trace=trace, extended_trace=trace)
end

function test()
    n = 0x5
    popt, f = factory(0.1, 1.3, 58.1, 0.5, n)
    @time res = optimize(f, 0.0, 10n, rel_tol=1e-2, show_trace=true, extended_trace=true)
    println("mode = ", Optim.minimizer(res))
    @time res = optimize(f, 0.0, 10n, rel_tol=1e-2, show_trace=true, extended_trace=true)
    @time res = solve(0.1, 1.3, 58.1, 0.5, n)
end

end # module MaxPlugin

# MaxPlugin.test()

# Local Variables:
# compile-command:"julia maxplugin.jl"
# End:
