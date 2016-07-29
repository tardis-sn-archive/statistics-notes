using JuMP
using NLopt
using Polynomials

θ = [1, -2, 3., 4]
p = Poly(θ)

m = Model(solver=NLoptSolver(algorithm=:LD_MMA, xtol_rel=1e-12, maxeval=15))

@variable(m, 0 <= x <= 1)
@NLobjective(m, Min, sum{θ[i] * x^(i-1), i=1:length(θ)})

setvalue(x, 0.5)
print(m)

@time status = solve(m)
#@time status = solve(m)
println("got ", getobjectivevalue(m), " at ", getvalue(x))

# Local Variables:
# compile-command:"julia minpoly.jl"
# End:
