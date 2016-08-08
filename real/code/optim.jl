module optim

using tardis
using DataFrames
using MaxPlugin
using NLopt
using Optim
using Plots; gr() #pyplot()
using Polynomials
using PyCall

function problem(;run=11, npackets=typemax(Int64), αOrder=2, βOrder=1)
    raw_data = readdata(run, npackets=npackets)
    frame = filter_positive(raw_data...)
    transform_data!(frame)
    return frame, targetfactory(frame, αOrder, βOrder)
end

function nlopt_factory(f, ∇f!)
    function(x::Vector, grad::Vector)
        if length(grad) > 0
            ∇f!(x, grad)
        end
        f(x)
    end
    end

    function trynlopt()
        αOrder = 1
        βOrder = 1
        dim = αOrder + βOrder
        frame, (f, ∇f!, Hf!) = problem(αOrder=αOrder, βOrder=βOrder, run=99)

        ## histogram(frame[:energies], normalized=true)
        ## α = 1.331; β=58.3
        ## plot!(x -> exp(loggamma(x, α, β)), linspace(0, 0.2, 500))
        ## Plots.pdf("optim.pdf")
        ## return

        myfunc = nlopt_factory(f, ∇f!)

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
            return minvalue - zmin
        end

        # GN_DIRECT_L, GN_ORIG_DIRECT_L
        # LN_BOBYQA, LN_COBYLA
        # LD_LBFGS, LD_MMA
        opt = Opt(:LN_COBYLA, dim)
        lowerbounds = zeros(dim)
        upperbounds = zeros(dim)
        lowerbounds[1] = 1.0 # α₀
        lowerbounds[2:end] = -50.0
        lowerbounds[αOrder + 1] = 0.0 # β₀
        upperbounds = -1 * lowerbounds
        upperbounds[1] = 4
        upperbounds[αOrder + 1] = 200
        ## lower_bounds!(opt, [1.0, 0.0])
        ## upper_bounds!(opt, [4.0, 100.0])
        xtol_rel!(opt, 1e-5)
        # ftol_rel!(opt, 1e-7)
        ## maxeval!(opt, 2500)
        max_objective!(opt, myfunc)

        θ = zeros(dim)
        θ[1] = 1.43 # α₀
        θ[αOrder + 1] = 43.2 # β₀

        αPoly = Poly(Vector{typeof(θ[1])}(αOrder))
        inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, αPoly, 1.0, 0))

        βPoly = Poly(Vector{typeof(θ[1])}(βOrder))
        inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, βPoly, 0.0, αOrder))

        @time minf, minx, ret = NLopt.optimize(opt, θ)
        println("got $minf at $minx (returned $ret)")

        ∇res, Hres = allocations(dim)
        Hf!(minx, Hres)
        evidence = laplace(minf, Hres)
        println("evidence = $evidence")

        # reset to normalized version of f
        f, ∇f!, Hf! = targetfactory(frame, αOrder, βOrder, evidence=evidence)
        myfunc = nlopt_factory(f, ∇f!)
        max_objective!(opt, myfunc)
        θ[1] = 1.52
        @time minf, minx, ret = NLopt.optimize(opt, θ)
        println("got $minf at $minx (returned $ret)")
        Hf!(minx, Hres)
        evidence = laplace(minf, Hres)
        println("evidence = $evidence")
    end

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
            init[1] = 1.43 # α₀
            init[αOrder + 1] = 43.2 # β₀
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

        return NLopt.optimize(opt, init)
    end

# """

# Compute contribution to `res` for `N`. Return updated `res` and bool
# if search should continue in this direction.

# """
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
* `a::Real`: power of the prior on the Poisson, 0: uniform, 1/2: Jeffrey's, 1:Jaynes
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

function main()
    # define order of linear models
    αOrder = 1
    βOrder = 1

    # read in data and create posterior
    dim = αOrder + βOrder
    ∇res, Hres = allocations(dim)
    frame, (P, ∇P!, HP!) = problem(αOrder=αOrder, βOrder=βOrder, run=99)

    # compute the posterior mode to estimate the evidence
    @time maxP, mode, ret = run_nlopt(frame, P, ∇P!, HP!, αOrder, βOrder, xtol_rel=1e-8)
    HP!(mode, Hres)
    evidence = laplace(maxP, Hres)
    println("got $maxP at $mode (returned $ret) and evidence $evidence")

    # the posterior mean with the normalized posterior
    pm = PosteriorMean(0.1, 0.2, 6)
    Pmean, ∇Pmean!, HPmean! = targetfactory(frame, αOrder, βOrder, evidence=evidence, pm=pm)
    println(Pmean(mode))
    @time maxPmean, mode, ret = run_nlopt(frame, Pmean, ∇Pmean!, HPmean!, αOrder, βOrder,
                                          init=mode, xtol_rel=1e-8)
    println("max of normalized posterior = $maxPmean at $mode")

    nb = NegBinom(5, 0.5)
    Pall, ∇Pall!, HPall! = targetfactory(frame, αOrder, βOrder, evidence=evidence, pm=pm, nb=nb)
    @time maxPall, mode, ret = run_nlopt(frame, Pall, ∇Pall!, HPall!, αOrder, βOrder,
                                          init=vcat(mode, 10), xtol_rel=1e-8, nb=nb)
    println("max of normalized all posterior = $maxPall at $mode")

    return
    points = collect(linspace(0.0, 0.6, 25))
    results = zeros(points)
    # skip X = 0
    optimize = false
    for i in 2:length(points)
        pm.X = points[i]
        results[i] = predict_small(frame, Pmean, ∇Pmean!, HPmean!, pm, αOrder, βOrder, mode, nb, optimize=optimize)
    end

    # self-normalize estimates through Simpson's rule
    @pyimport scipy.integrate as si
    norm = si.simps(results, points)
    println("norm = $norm")
    results /= norm
    println(results)

    Plots.plot(points, results)
fname =
    Plots.pdf("test_$(optimize? "opt" : "noopt").pdf")

    # # update X
    # pm.N = 7
    # println(Pmean(mode))
    # @time maxPmean, mode, ret = run_nlopt(frame, Pmean, ∇Pmean!, HPmean!, αOrder, βOrder,
    #                                       init=mode, xtol_rel=1e-8)
    # println("max of normalized posterior = $maxPmean at $mode")

end
end # module optim

# trynlopt()
optim.main()

# predict([0.1, 0.2, 0.3, 0.4], run=9)

# Local Variables:
# compile-command:"julia optim.jl"
# End:
