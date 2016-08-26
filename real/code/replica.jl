push!(LOAD_PATH, pwd())
using tardis
using Base.Test
import ConjugatePriors
import Distributions
using DataFrames
using HDF5
using Optim
using Plots
import ForwardDiff
chunk = ForwardDiff.Chunk{3}()

function plotprediction(sums, predict, stddev, i=1)
    pyplot()
    Plots.histogram(sums, normed=true, lab="$(length(sums)) replicas")
    vline!([sums[i]], line=:red, lab="lum. in replica $i")
    dist=Distributions.Normal(predict[i], stddev[i])
    plot!(x->Distributions.pdf(dist, x), linspace(predict[i] - 4*stddev[i], predict[i] + 4*stddev[i]), lab="prediction from replica $i", line=:black)
    Plots.pdf("replica-prediction.pdf")
end

function read_real(nsim)
    n = Array(Int64, (nsim,))
    x = Array(Float64, (nsim,))
    means = zeros(x)
    sumsqdiff = zeros(x)

    νmin, νmax = 1.00677, 1.023018

    function scale_data!(frame::DataFrame, nuscale::Float64=1e15, xscale::Float64=1e38)
        sort!(frame, cols=:nus)

        # reference to the frame if we index directly
        # things like en /= 2 operate on a copy of the data
        frame[:nus] /= nuscale
        frame[:energies] /= xscale
    end

    function read_run(run)
        raw_data = readdata(run)
        frame = filter_positive(raw_data...)
        scale_data!(frame)
        nmin=searchsortedfirst(frame[:nus], νmin)
        nmax=searchsortedlast(frame[:nus], νmax)
        n[run] = nmax - nmin + 1
        x[run] = sum(frame[:energies][nmin:nmax])
        means[run] = mean(frame[:energies][nmin:nmax])
        # remove Bessel's correction
        sumsqdiff[run] = (n[run]-1) * var(frame[:energies][nmin:nmax])
    end

    for i in 1:nsim
        read_run(i)
    end

    n, x, means, sumsqdiff
end

function read_virtual(nsim; νmin=1.00677e15, νmax=1.023018e15, cutoff=1e-20)
    n = Array(Int64, (nsim,))
    x = Array(Float64, (nsim,))
    means = zeros(x)
    sumsqdiff = zeros(x)

    h5open("/data/tardis/virtual_packets.h5", "r") do f
        for (i, id) in enumerate(names(f["data"]))
            (i > nsim) && break

            # filter frequencies in bin
            nus = f["/data/$(id)/runner_virt_packet_nus/values"][:]
            mask = (nus .> νmin) & (nus .< νmax)
            # select the energies in the frequency bin
            energies = f["/data/$(id)/runner_virt_packet_energies/values"][:][mask]
            # remove small values
            energies = energies[energies .> cutoff]
            n[i] = length(energies)
            x[i] = sum(energies)
            means[i] = mean(energies)
            sumsqdiff[i] = (n[i]-1)*var(energies)
        end
    end
    n, x, means, sumsqdiff
end

@enum TARGET Posterior FirstMoment SecondMoment

function targetfactory(target::TARGET, n::Int64, xmean::Float64, xsumsq::Float64, a::Float64=1)
    0 <= a <= 1 || error("need a in [0,1]")

    function logposterior(x::Vector)
        λ, μ, σSq = x

        # gamma(λ), set β=1
        α = n-a+1
        resgamma = - lgamma(α) + (α-1)*log(λ) - λ
        ## println(resgamma)

        # normal-inversegamma
        d = ConjugatePriors.NormalInverseGamma(xmean, n, n/2, xsumsq/2)

        resnormal = -1/2 * (log(2*π*σSq / d.v0) + (μ-d.mu)^2 * d.v0 / σSq)
        resinvgamma = -lgamma(d.shape) + d.shape * log(d.scale) - (d.shape + 1) * log(σSq) - d.scale / σSq

        ## println(resinvgamma," ", resnormal)
        return resgamma + resnormal + resinvgamma
    end

    """log(-d/dt f(target=0|x))"""
    function logfirst(x::Vector)
        λ, μ, σSq = x
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) * sqrt(λ * σSq / (2 * π)) + 1/2 * λ * μ * (1 + erf(sqrt(exponent)))
        log(tmp)
    end

    """log(d²/dt² f(target=0|x))"""
    function logsecond(x::Vector)
        λ, μ, σSq = x
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) * sqrt(2 / π * λ^3 * σSq) * μ + λ*(λ*μ^2 + σSq)*(1 + erf(sqrt(exponent)))
        log(1/2 * tmp)
    end

    if target === Posterior
        return -logposterior
    elseif target === FirstMoment
        return x -> -(logposterior(x) + logfirst(x))
    elseif target === SecondMoment
        return x -> -(logposterior(x) + logsecond(x))
    end
end


function test()
    n, sums, means, sumsqdiff = read_real(10)
    i = 10
    logposterior=logposterior_factory(n[i], means[i], sumsqdiff[i])
    target(x) = -logposterior(x)
    x = [501.0,1.406,0.001]
    res = optimize(target, x, Newton(), OptimizationOptions(autodiff = true))
    # compare to mathematica
    @test -Optim.minimum(res) ≈ 10.515182417513643

    H = zeros((3,3))
    # measured, chunk size of 3 is twice as fast as 1 or 2
    ForwardDiff.hessian!(H,logposterior, φ, chunk)

    # need a really good starting point, else run into
    # ERROR: DomainError:
    # in logposterior at none:12

    # minimize the negative to maximize
    target(x) = -logposterior(x) - logfirst(x)
    res = optimize(target, x, Newton(), OptimizationOptions(autodiff=true))
    @test -Optim.minimum(res) ≈ 17.06965251449796

    ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
    logmean = tardis.laplace(-Optim.minimum(res), H)
    @test exp(logmean) ≈ 697.496

    target(x) = -logposterior(x) - logsecond(x)
    res = optimize(target, x, Newton(), OptimizationOptions(autodiff=true))
    @test -Optim.minimum(res) ≈ 23.626124437003362
    logsecmom = tardis.laplace(-Optim.minimum(res), H)
    # difference in 5th decimal, perhaps because of rounding in determinant?
    @test_approx_eq_eps logsecmom 13.104967686073357 2e-3
    stddev = sqrt(exp(logsecmom) - exp(logmean)^2)

    ## grad = zeros(3)
    ## ForwardDiff.gradient!(grad, logposterior, φ, chunk)
    ## ForwardDiff.gradient!(grad, x::Vector -> logfirst(x) + logposterior(x), φ, chunk)
    ## ForwardDiff.gradient!(grad, x::Vector -> logsecond(x) + logposterior(x), φ, chunk)
end
# can'target use with autodiff: Distributions expects Float64
## posteriorμσSq = ConjugatePriors.NormalInverseGamma(x[i], n[i], n[i]/2, sumsq[i]/2)
## posteriorλ = Distributions.Gamma(n[i], 1.0)
## ConjugatePriors.logpdf(posteriorμσSq, 697, 0.000947)
## Distributions.logpdf(posteriorλ, 500)
## log_posterior(x::Vector) = Distributions.logpdf(posteriorλ, x[1]) + ConjugatePriors.logpdf(posterior, x[2], x[3])

# TODO safe Hessian in buffer passed by user
function uncertainty(n::Int64, xmean::Float64, xsumsq::Float64, a::Float64=1.0)
    # initial guess for optimization: slightly off to avoid
    # ERROR: Linesearch failed to converge
    init = [n * 1.0001, xmean * 1.0005, xsumsq / n * 1.0005]
    # Hessian
    H = Array{Float64}(3,3)

    # mean
    target = targetfactory(FirstMoment, n, xmean, xsumsq, a)
    res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
    ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
    logfirst = tardis.laplace(-Optim.minimum(res), H)
    mean = exp(logfirst)

    # second moment
    target = targetfactory(SecondMoment, n, xmean, xsumsq, a)
    res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
    ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
    logsecond = tardis.laplace(-Optim.minimum(res), H)

    mean, sqrt(exp(logsecond) - mean^2)
end

function analyze(nsim, readf)
    n, sums, means, sumsqdiff = readf(nsim)

    predict, stddev = begin
        a = zeros(means)
        b = zeros(means)

        for i in 1:nsim
            a[i], b[i] = uncertainty(n[i], means[i], sumsqdiff[i])
            println("$(i): $(a[i]), $(b[i])")
        end
        a, b
    end
    println("Observed: $(mean(sums)) ± $(std(sums))")
    println("Predicted: $(mean(predict)) ± $(mean(stddev))")

    plotprediction(sums, predict, stddev)

    n, sums, predict, stddev
end
