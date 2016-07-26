include("tardis.jl")
using tardis
using DataFrames
using Optim
using NLopt
using Polynomials

function problem(;run=11, αOrder = 2, βOrder = 1)
    raw_data = readdata(run)
    frame = filter_positive(raw_data...)
    transform_data!(frame)
    return frame, targetfactory(frame, αOrder, βOrder)
end

function tryoptim()
    frame, f, ∇f!, Hf! = problem()

    # initial value
    θ = [1.3, 0, 54]

    res = Optim.optimize(
                   f, θ, NelderMead(),
                   ## f, ∇f!, θ, BFGS(),
                   ## f, ∇f!, Hf!, θ, Newton(),
                   OptimizationOptions(iterations=1000))

    println(res)
    θ = Optim.minimizer(res)
    println("f($(θ)) = $(f(θ))")
end

function trynlopt()
    αOrder = 2
    βOrder = 1
    dim = αOrder + βOrder
    frame, (f, ∇f!, Hf!) = problem(αOrder=αOrder, βOrder=βOrder)

    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            ∇f!(x, grad)
        end

        return f(x)
    end

    """
    Prior constraint for α > 1 and β > 0 in the form g(θ) ≤ 0
    """
    function myconstraint(x::Vector, grad::Vector, frame::DataFrame, p::Poly,
                          minvalue::Float64, offset::Int64)
        ν::Vector{Float64} = frame[:nus]
        z::Float64 = 0
        zmin::Float64 = oftype(1.0, Inf)
        imin::Int64 = 0

        for i in eachindex(ν)
            @inbounds z = polyval(p, ν[i])
            if z < zmin
                zmin = z
                imin = i
            end

        end
        if length(grad) > 0
            for m in 1:length(p)
                grad[offset + m] = -ν[imin]^(m-1)
            end
        end
        return 1.0 - zmin
    end

    opt = Opt(:LD_MMA, dim)
    xtol_rel!(opt, 1e-4)
    min_objective!(opt, myfunc)

    θ = zeros(dim)
    θ[1] = 1.5 # α₀
    θ[αOrder + 1] = 50 # β₀

    αPoly = Poly(Vector{typeof(θ[1])}(αOrder))
    inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, αPoly, 1.0, 0))

    βPoly = Poly(Vector{typeof(θ[1])}(βOrder))
    inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, βPoly, 0.0, αOrder))

    minf, minx, ret = NLopt.optimize(opt, θ)
    println("got $minf at $minx (returned $ret)")
end

trynlopt()
