module optim

using tardis
using DataFrames
using Optim
using NLopt
using Plots;
using Polynomials

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

    function run_nlopt(frame, f,  ∇f!, Hf!, αOrder, βOrder; init=false, xtol_rel=1e-5)
        T = Float64
        dim = αOrder + βOrder

        myfunc = nlopt_factory(f, ∇f!)

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
        lower_bounds!(opt, lowerbounds)
        upper_bounds!(opt, upperbounds)
        xtol_rel!(opt, xtol_rel)
        max_objective!(opt, myfunc)

        # initial value
        if init === false
            init = zeros(dim)
            init[1] = 1.43 # α₀
            init[αOrder + 1] = 43.2 # β₀
        else
            @assert(length(init) == dim)
        end

        αPoly = Poly(Vector{T}(αOrder))
        inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, αPoly, 1.0, 0))

        βPoly = Poly(Vector{T}(βOrder))
        inequality_constraint!(opt, (x, g) -> myconstraint(x, g, frame, βPoly, 0.0, αOrder))

        return NLopt.optimize(opt, init)
    end

    function predict(ν::Real, X::Vector; run=9, αOrder=1, βOrder=1)

        # now predict

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
        pm = PosteriorMean(0.1, 0.2, 10)
        Pmean, ∇Pmean!, HPmean! = targetfactory(frame, αOrder, βOrder, evidence=evidence, pm=pm)
        println(Pmean(mode))
        @time maxPmean, mode, ret = run_nlopt(frame, Pmean, ∇Pmean!, HPmean!, αOrder, βOrder,
                                        init=mode, xtol_rel=1e-8)
        println("max of normalized posterior = $maxPmean at $mode")

        # update X
        pm.X = 2.0
        println(Pmean(mode))
        @time maxPmean, mode, ret = run_nlopt(frame, Pmean, ∇Pmean!, HPmean!, αOrder, βOrder,
                                        init=mode, xtol_rel=1e-8)
        println("max of normalized posterior = $maxPmean at $mode")

    end
end # module optim

# trynlopt()
optim.main()

    # predict([0.1, 0.2, 0.3, 0.4], run=9)

# Local Variables:
# compile-command:"julia optim.jl"
# End:
