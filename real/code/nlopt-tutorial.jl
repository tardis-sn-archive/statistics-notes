using JuMP
using NLopt

"""

Minimize |g(x)| where, g(x) = (x-5)²-1. The two degenerate solutions
are x=5±1 where g(x)=0. Reformulate with a dummy variable and two nonlinear
constraints to preserve the differentiability everywhere following
http://ab-initio.mit.edu/wiki/index.php/NLopt_Introduction#Equivalent_formulations_of_optimization_problems
"""

# only some NLopt solvers support nonlinear constraints: GN_DIRECT[_L], LD_MMA, ISRES
# If another one is chosen, the error msg. is not very helpful
#
# ERROR: LoadError: ArgumentError: invalid NLopt arguments
m = Model(solver=NLoptSolver(algorithm=:LD_MMA))

# adding box constraints helps
@variable(m, 0 <= t <= 100)
@variable(m, 0 <= x <= 20)


@NLobjective(m, Min, t)
@NLconstraint(m, (x-5)^2 -1 -t <= 0)
@NLconstraint(m, -(x-5)^2 +1 -t <= 0)

setvalue(t, 5)
setvalue(x, 1) # if x too large, MMA does not converge to the right solution

print(m)

status = solve(m)

println("got ", getobjectivevalue(m), " at ", [getvalue(t),getvalue(x)])
