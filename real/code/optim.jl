module optim

using ..Tardis
using DataFrames
using MaxPlugin
using MinPoly
using NLopt
using Optim
using Plots
using Polynomials

function problem(;run=11, npackets=typemax(Int64), αOrder=2, βOrder=1)
    raw_data = readdata(run, npackets=npackets)
    frame = filter_positive(raw_data...)
    transform_data!(frame)
    return frame, targetfactory(frame, αOrder, βOrder)
end

nlopt_factory(f, ∇f!) = function(x::Vector, grad::Vector)
    if length(grad) > 0
        ∇f!(x, grad)
    end
    f(x)
end

"""
# Arguments
* nb: controls what the target density is: if `nothing`, target = likelihood or posterior mean, if given with valied `n` and `a`, optimize over N, treating it as continuous
"""
function run_nlopt(frame, f,  ∇f!, Hf!, αOrder, βOrder; init=nothing, xtol_rel=1e-5, nb=nothing)
    T = Float64
    dim = αOrder + βOrder
    if nb !== nothing
        dim += 1
    end

    myfunc = nlopt_factory(f, ∇f!)

    # todo replace with minpoly2
    """
    Prior constraint for α > 1 and β > 0 in the form g(θ) ≤ 0
    """
    function myconstraint(x::Vector, grad::Vector, frame::DataFrame, p::Poly,
                          minvalue::Float64, offset::Int64)
        ν::Vector{T} = frame[:nus]
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
        return minvalue - zmin
    end

    # opt = Opt(:LD_MMA, dim)
    # opt = Opt(:LD_SLSQP, dim)
    opt = Opt(:LN_COBYLA, dim)
    lowerbounds = zeros(dim)
    upperbounds = zeros(dim)
    lowerbounds[1] = 1.0 # α₀
    lowerbounds[2:end] = -50.0
    lowerbounds[αOrder + 1] = 0.0 # β₀
    upperbounds = -1 * lowerbounds
    upperbounds[1] = 4
    upperbounds[αOrder + 1] = 200
    if nb !== nothing
        lowerbounds[end] = 0.0
        upperbounds[end] = 10 * nb.n
    end

    lower_bounds!(opt, lowerbounds)
    upper_bounds!(opt, upperbounds)
    xtol_rel!(opt, xtol_rel)
    max_objective!(opt, myfunc)

    # initial value
    if init === nothing
        init = zeros(dim)
        init[1] = 1.6 # α₀
        init[αOrder + 1] = 53.19 # β₀
        if nb !== nothing
            init[end] = nb.n
        end
    else
        @assert(length(init) == dim)
    end

    αPoly = Poly(Vector{T}(αOrder))
    inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, αPoly, 1.0, 0))

    βPoly = Poly(Vector{T}(βOrder))
    inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, βPoly, 0.0, αOrder))

    res = NLopt.optimize(opt, init)
    # println("α constraint at the mode ", myconstraint(res[2], [], frame, αPoly, 1.0, 0))
    # println("β constraint at the mode ", myconstraint(res[2], [], frame, βPoly, 0.0, αOrder))
    return res
end

"""

Compute contribution to `res` for `N`. Return updated `res` and bool
if search should continue in this direction.

"""
function search_step!(frame::DataFrame, Pmean, ∇Pmean!, HPmean!, Hres::Matrix, pm::PosteriorMean,
                      αOrder::Integer, βOrder::Integer, nb::NegBinom, res::Real, θ::Vector, ε::Real;
                      xtol_rel=1e-5, optimize=true)
    # N>0 required to search
    if pm.N <= 0
        return res, false
    end

    # optimize and Laplace integrate
    if optimize
        maxPmean, mode, ret = run_nlopt(frame, Pmean, ∇Pmean!, HPmean!, αOrder, βOrder,
                                        init=θ, xtol_rel=xtol_rel)
    else
        mode = θ
        maxPmean = Pmean(mode)
    end
    HPmean!(θ, Hres)
    log_integral = laplace(maxPmean, Hres)

    # copy back mode
    θ[:] = mode

    # have to leave the log scale now
    latest = exp(lognegativebinomial(pm.N, nb.n, nb.a) + log_integral)
    res += latest

    # println("$(pm): latest=$latest, res=$res")

    return res, (latest / res) > ε
end

"""

# Arguments
* `Pmean`: posterior mean function, properly normalized.
* `pm`: data used in `Pmean` et al.
* `a::Real`: power of the prior on the Poisson, 0: uniform, 1/2: Jeffreys, 1:Jaynes
* `θ::Vector`: the mode of the posterior that is used to compute the evidence
* `Ninit::Integer`: initial value of `N`. If `N>0`, the computation starts here.
  If `N==0`, an initial is computed from optimizing the negative binomial x posterior predictive,
  with `α` and `β` evaluated at `θ`
* `optimize=true`: Integrate over parameters via Laplace requiring optimization. If `false`, use parameters at the mode w/o re-optimizing.
"""
function predict_small(frame::DataFrame, Pmean, ∇Pmean!, HPmean!, pm::PosteriorMean,
                       αOrder::Integer, βOrder::Integer,
                       θ::Vector, nb::NegBinom;
                       ε::Real=5e-3, Ninit::Integer=0, xtol_rel=1e-5, optimize=true)
    @assert length(θ) == αOrder + βOrder

    # initialization
    res = 0.0
    if Ninit == 0
        # perform optimization, optim only does minimization but we
        # actually optimize to find the N that will likely give the
        # largest contribution. It's only approximate but should be a
        # good starting value

        # eval polynomials at mode
        p = Poly(θ[1:αOrder])
        α = p(pm.ν)
        p = Poly(θ[αOrder+1:αOrder+βOrder])
        β = p(pm.ν)

        res = MaxPlugin.solve(pm.X, α, β, nb.a, nb.n)
        Ninit = convert(Integer, ceil(Optim.minimizer(res)))
    end
    @assert(Ninit >= 1)
    println("initial N=$Ninit")

    Nup = Ninit
    Ndown = Ninit
    θup, θdown = copy(θ),copy(θ)
    _, Hres = allocations(length(θ))
    res = 0.0

    # bind most arguments
    search(N) = begin
        pm.N = N
        search_step!(frame, Pmean, ∇Pmean!, HPmean!, Hres, pm, αOrder, βOrder,
                    nb, res, θ, ε, xtol_rel=xtol_rel, optimize=optimize)
        end

    res, _ = search(Ninit)

    goup, godown = true, true

    while goup || godown
        if goup
            Nup += 1
            pm.N = Nup
            res, goup = search(Nup)
        end
        if godown
            Ndown -= 1
            pm.N = Ndown
            res, godown = search(Ndown)
        end
    end
    return res
end

end # module optim

# trynlopt()

# predict([0.1, 0.2, 0.3, 0.4], run=9)

# Local Variables:
# compile-command:"julia optim.jl"
# End:
