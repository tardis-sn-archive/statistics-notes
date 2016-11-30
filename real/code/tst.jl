import optim
import PyCall

function main()
    # define order of linear models
    αOrder = 1
    βOrder = 1

    # read in data and create posterior
    dim = αOrder + βOrder
    ∇res, Hres = optim.allocations(dim)
    frame, (P, ∇P!, HP!) = optim.problem(αOrder=αOrder, βOrder=βOrder, run=99)

    # compute the posterior mode to estimate the evidence
    @time maxP, mode, ret = optim.run_nlopt(frame, P, ∇P!, HP!, αOrder, βOrder, xtol_rel=1e-3)
    HP!(mode, Hres)
    evidence = optim.laplace(maxP, Hres)
    println("got $maxP at $mode (returned $ret) and evidence $evidence")

    # the posterior mean with the normalized posterior
    pm = optim.PosteriorMean(0.1, 0.2, 6.0)
    Pmean, ∇Pmean!, HPmean! = optim.targetfactory(frame, αOrder, βOrder, evidence=evidence, pm=pm)
    println(Pmean(mode))
    @time maxPmean, mode, ret = optim.run_nlopt(frame, Pmean, ∇Pmean!, HPmean!, αOrder, βOrder,
                                          init=mode, xtol_rel=1e-8)
    println("max of normalized posterior = $maxPmean at $mode")

    return

    nb = NegBinom(5, 0.5)
    Pall, ∇Pall!, HPall! = targetfactory(frame, αOrder, βOrder, evidence=evidence, pm=pm, nb=nb)
    @time maxPall, mode, ret = run_nlopt(frame, Pall, ∇Pall!, HPall!, αOrder, βOrder,
                                          init=vcat(mode, 10), xtol_rel=1e-8, nb=nb)
    println("max of normalized all posterior = $maxPall at $mode")
    # return

    points = collect(linspace(0.0, 0.6, 25))
    results = zeros(points)
    # skip X = 0
    optimize = false
    for i in 2:length(points)
        pm.X = points[i]
        results[i] = optim.predict_small(frame, Pmean, ∇Pmean!, HPmean!, pm, αOrder, βOrder, mode, nb, optimize=optimize)
    end

    # self-normalize estimates through Simpson's rule
    PyCall.@pyimport scipy.integrate as si
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

main()
