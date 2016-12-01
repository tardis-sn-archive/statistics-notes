module Replica

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

"""

"""
function plotprediction(sums, predict, stddev, i=1)
    Plots.histogram(sums, normed=true, lab="$(length(sums)) replicas")
    vline!([sums[i]], line=:red, lab="lum. in replica $i")
    dist=Distributions.Normal(predict[i], stddev[i])
    plot!(x->Distributions.pdf(dist, x), linspace(predict[i] - 4*stddev[i], predict[i] + 4*stddev[i]), lab="prediction from replica $i", line=:black)
    Plots.pdf("replica-prediction.pdf")
    # replot to show it interactively
    info("Replica $i: $(sums[i]) vs $(predict[i]) ± $(stddev[i])")
    plot!()
end

function read_real(nsim; npackets=typemax(Int64), νmin=1.00677, νmax=1.023018)
    n = Array(Int64, (nsim,))
    x = Array(Float64, (nsim,))
    means = zeros(x)
    sumsqdiff = zeros(x)

    function scale_data!(frame::DataFrame, nuscale::Float64=1e15, xscale::Float64=1e38)
        sort!(frame, cols=:nus)

        # reference to the frame if we index directly
        # things like en /= 2 operate on a copy of the data
        frame[:nus] /= nuscale
        frame[:energies] /= xscale
    end

    function read_run(run)
        raw_data = readdata(run; npackets=npackets)
        frame = filter_positive(raw_data...)
        # scale_data!(frame)
        transform_data!(frame)
        nmin=searchsortedfirst(frame[:nus], νmin)
        nmax=searchsortedlast(frame[:nus], νmax)
        n[run] = nmax - nmin + 1
        x[run] = sum(frame[:energies][nmin:nmax])
        means[run] = mean(frame[:energies][nmin:nmax]) # 1/n[run] * x[run]
        # remove Bessel's correction
        sumsqdiff[run] = (n[run]-1) * var(frame[:energies][nmin:nmax])
        # if run == 1
        #     info("first run: n=$(n[1]), sum=$(x[1]), mean=$(means[1]), sum^2=$(sumsqdiff[1])")
        #     println(frame[:energies][nmin:nmax])
        # end
    end

    for i in 1:nsim
        read_run(i)
    end

    n, x, means, sumsqdiff
end

function read_virtual(nsim; νmin::Float64=1.00677, νmax::Float64=1.023018, νscale::Float64=1e15,
                      cutoff::Float64=1e-20, filename::String="virtual_packets.h5")
    n = Array(Int64, (nsim,))
    x = Array(Float64, (nsim,))
    means = zeros(x)
    sumsqdiff = zeros(x)

    h5open(filename, "r") do f
        for (i, id) in enumerate(names(f["data"]))
            (i > nsim) && break

            # filter frequencies in bin
            nus = f["/data/$(id)/runner_virt_packet_nus/values"][:]
            mask = (nus .> νmin*νscale) & (nus .< νmax*νscale)
            # select the energies in the frequency bin
            energies = f["/data/$(id)/runner_virt_packet_energies/values"][:][mask]
            # remove small values
            energies = energies[energies .> cutoff]
            n[i] = length(energies)
            n[i] > 0 || error("No packets found in [$νmin, $νmax]")
            x[i] = sum(energies)
            means[i] = mean(energies)
            sumsqdiff[i] = (n[i]-1)*var(energies)
        end
    end
    n, x, means, sumsqdiff
end

"""
Enumeration to specify which target function to create
"""
@enum TARGET Posterior FirstMoment SecondMoment ThirdMoment FourthMoment

"""
Create a function for Laplace approximation
"""
function targetfactory(target::TARGET, n::Int64, xmean::Float64, xsumsq::Float64, a::Float64=1)
    0 <= a <= 1 || error("need a in [0,1]")
    (xmean > 0) || error("nonpositive xmean $xmean")
    (xsumsq > 0) || error("nonpositive xsumsq $xsumsq")

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

    """log(-d^3/dt^3 f(target=0|x))"""
    function logthird(x::Vector)
        λ, μ, σSq = x
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) / sqrt(2π) * λ^(3/2) * sqrt(σSq) * (λ*μ^2 + 2σSq)
        tmp += 1/2 * λ * μ * (λ*μ^2 + 3σSq)*(1 + erf(sqrt(exponent)))
        log(tmp)
    end

    """log(d^4/dt^4 f(target=0|x))"""
    function logfourth(x::Vector)
        λ, μ, σSq = x
        exponent = λ*μ^2 / (2*σSq)
        tmp = exp(-exponent) / sqrt(2π) * λ^(5/2) * μ * sqrt(σSq) * (λ*μ^2 + 5σSq)
        tmp += 1/2 * λ^2 * (λ^2*μ^4 + 6λ*μ^2 + 3σSq)*(1 + erf(sqrt(exponent)))
        log(tmp)
    end

    # optimizer minimizes => add minus sign at the last possible moment
    if target === Posterior
        return x -> -logposterior(x)
    elseif target === FirstMoment
        return x -> -(logposterior(x) + logfirst(x))
    elseif target === SecondMoment
        return x -> -(logposterior(x) + logsecond(x))
    elseif target === ThirdMoment
        return x -> -(logposterior(x) + logthird(x))
    elseif target === FourthMoment
        return x -> -(logposterior(x) + logfourth(x))
    end
end

# TODO safe Hessian in buffer passed by user
function uncertainty(n::Int64, xmean::Float64, xsumsq::Float64, a::Float64=1.0)
    # initial guess for optimization: slightly off to avoid
    # ERROR: Linesearch failed to converge
    init = [n * 1.0001, xmean * 1.0005, xsumsq / n * 1.0005]
    # Hessian
    H = Array{Float64}(3,3)

    function integrate(moment::TARGET)
        target = targetfactory(moment, n, xmean, xsumsq, a)
        res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
        ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
        exp(tardis.laplace(-Optim.minimum(res), H))
    end

    # mean
    μ = integrate(FirstMoment)

    # second moment
    # target = targetfactory(SecondMoment, n, xmean, xsumsq, a)
    # res = optimize(target, init, Newton(), OptimizationOptions(autodiff=true))
    # ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
    # logsecond = tardis.laplace(-Optim.minimum(res), H)
    second = integrate(SecondMoment)
    σ = sqrt(second - μ^2)

    third = integrate(ThirdMoment)
    skewness = (third - 3μ*second + 2μ^3) / σ^3

    fourth = integrate(FourthMoment)
    kurtosis = (fourth - 4μ*third + 6μ^2*second - 4μ^4) / σ^4

    # excess kurtosis = kurtosis - 3
    return μ, σ, skewness, kurtosis - 3
end

"""
Analyze all replicas and compare mean and standard deviation from the replicas and the model
"""
function analyze(nsim, readf)
    n, sums, means, sumsqdiff = readf(nsim)

    predict, stddev, skewness, excess_kurtosis = begin
        a = zeros(means)
        b = zeros(means)
        c = zeros(means)
        d = zeros(means)

        for i in 1:nsim
            # println("run $i")
            a[i], b[i], c[i], d[i] = uncertainty(n[i], means[i], sumsqdiff[i])
           # println("$(i): $(a[i]), $(b[i])")
        end
        a, b, c, d
    end
    println("Observed: $(mean(sums)) ± $(std(sums))")
    println("Predicted: $(mean(predict)) ± $(mean(stddev))")

    n, sums, predict, stddev, skewness, excess_kurtosis
end

function test()
    n, sums, means, sumsqdiff = read_real(10)
    i = 10
    a = 1.0
    target = targetfactory(Posterior, n[i], means[i], sumsqdiff[i], a)
    # target(x) = -logposterior(x)
    x = [501.0,1.406,0.001]
    res = optimize(target, x, Newton(), OptimizationOptions(autodiff = true))
    # compare to mathematica
    @test -Optim.minimum(res) ≈ 10.515182417513643

    H = zeros((3,3))
    # measured, chunk size of 3 is twice as fast as 1 or 2
    ForwardDiff.hessian!(H,target, Optim.minimizer(res), chunk)

    println("Posterior")
    println(res)

    # need a really good starting point, else run into
    # ERROR: DomainError:
    # in logposterior at none:12

    # minimize the negative to maximize
    target = targetfactory(FirstMoment, n[i], means[i], sumsqdiff[i], a)
    res = optimize(target, x, Newton(), OptimizationOptions(autodiff=true))
    @test -Optim.minimum(res) ≈ 17.06965251449796

    ForwardDiff.hessian!(H, target, Optim.minimizer(res), chunk)
    logmean = tardis.laplace(-Optim.minimum(res), H)
    @test exp(logmean) ≈ 697.496291
    println("First moment")
    println(res)

    target = targetfactory(SecondMoment, n[i], means[i], sumsqdiff[i], a)
    res = optimize(target, x, Newton(), OptimizationOptions(autodiff=true))
    @test -Optim.minimum(res) ≈ 23.626124437003362
    logsecmom = tardis.laplace(-Optim.minimum(res), H)
    # difference in 5th decimal, perhaps because of rounding in determinant?
    @test_approx_eq_eps logsecmom 13.104967686073357 2e-3
    stddev = sqrt(exp(logsecmom) - exp(logmean)^2)

    println("Second moment")
    println(res)
end

end
