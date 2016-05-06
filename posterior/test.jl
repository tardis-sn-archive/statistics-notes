using HDF5
using DataFrames

using Plots;
# gadfly()
pyplot()
# gr()

# using Mamba
using Polynomials
using Optim

## h5open("real_tardis_250.h5", "r") do file
##     ## dump(file)
##     file["energies"]
##     nus = file["nus"]
##     print(length(energies[1,:]))
##     histogram(energies[:,run])
##     show()
##     Plots.pdf("testPlots.pdf")
## end

function read_data(runid=10, filename="real_tardis_250.h5")
    h5open(filename, "r") do file
        # read only a single run
        file["nus"][:,runid], file["energies"][:,runid]
    end
end

"""
    filter_positive(x, y)

For two arrays of same length, filter out elements with negative x and
return a data frame with columns (nu, energies)

"""

function filter_positive(nus, energies)
    assert(length(energies) == length(nus))

    mask = (energies .> 0.0)
    DataFrame(nus=nus[mask], energies=energies[mask])

    # can't trip result of zip as a collection
    ## filter((x, y) -> y > 0, zip(nus, energies))

    ## # works but yields a subarray that is not so convenient to work with
    ## mask = (energies .> 0.0)
    ## # iterate over columns => few rows, many columns to be cache friendly
    ## x = Array(typeof(energies[1]), 2, length(mask))
    ## pos = 0
    ## for i in eachindex(mask)
    ##     if mask[i]
    ##         pos += 1
    ##         x[1, pos] = nus[i]
    ##         x[2, pos] = energies[i]
    ##     end
    ## end
    ## sub(x,:, 1:pos)


    # CRITICAL take type of first element, assumed to be scalar!
    ## x = Array(typeof(nus[1]), 2, length(energies))
    ## npos::Int64 = 0
    ## for i in eachindex(energies)
    ##     if energies[i] > 0
    ##         npos += 1
    ##         x[1, npos] = nus[i]
    ##         x[2, npos] = energies[i]
    ##     end
    ## end
    ## assert(npos > 0)
    ## return x[:,1:npos]
end

"""

Sort by frequency, then rescale and flip the energies, then rescale
the frequencies.

"""

function transform_data!(frame::DataFrame)
    sort!(frame, cols=:nus)

    # update energies

    # reference to the frame if we index directly
    # things like en /= 2 operate on a copy of the data
    en = frame[:energies]
    maxen = (1.0 + 1e-6) * maximum(en)
    for i in eachindex(en)
        en[i] = 1.0 - en[i] / maxen
    end

    # the last frequency is the largest
    frame[:nus] /= frame[:nus][end]

    println("min nu = $(frame[:nus][1]), max en = $(maximum(en))")

    # return nothing, just mutate input
    nothing
end

# function polynu(nu::Float64, x::Vector{Float64})
#     @evalpoly(nu, x)
# end

function logGamma(x::Float64, α::Float64, β::Float64)
  -α * log(β) - lgamma(α) + (α-1)*log(x) - x / β
end

function polynu(a::Vector, nu::Real)
    res = zero(a[1])
    power = one(a[1])
    for x in a
        res += x * power
        power *= nu
    end
end

function logtarget(frame::DataFrame)
    αOrder::Int32 = 1
    βOrder::Int32 = 1
    ν::Vector{Float64} = frame[:nus]
    x::Vector{Float64} = frame[:energies]

    # closure: needs frame::Dataframe in scope
    function log_likelihood(θ::Vector)
        res::Float64 = 0
        α::Float64 = 0
        β::Float64 = 0
        invalid = -1e100 # oftype(1.0, -Inf)
        # αvec = sub(θ, 1:αOrder)
        # βvec = sub(θ, αOrder+1:αOrder + βOrder)
        # αvec = θ[1:αOrder]
        # βvec = θ[αOrder+1:αOrder + βOrder]

        αPoly = Poly(Vector{typeof(θ[1])}(αOrder))
        βPoly = Poly(Vector{typeof(θ[1])}(βOrder))

        for i in eachindex(ν)
            for n in 1:αOrder
                @inbounds αPoly.a[n] = θ[n]
            end
            @inbounds α = polyval(αPoly, ν[i])

            for n in 1:βOrder
                @inbounds βPoly.a[n] = θ[αOrder + n]
            end
            @inbounds β = polyval(βPoly, ν[i])

            # α = polynu(αvec, ν[i])
            # β = polynu(βvec, ν[i])

            # prior: reject point if α or β too small
            if α < 1.0
                # println("α too small for packet $i")
                return invalid
            end

            if β < 0.0
                # println("β too small for packet $i")
                return invalid
            end

            res += logGamma(x[i], α, β)
            # if i == 1
            #     println("x = $(x[1]), ν = $(ν[1]), α = $α, β = $β, res = $res")
            # end
            if !isfinite(res)
                return invalid
            end
        end
        # suitable for minimization
       -res
    end
end

# function factory(nu::Vector{Float64}, x::Vector{Float64})
#   prior_alpha = [Uniform(1, 2), Uniform(-3, 3), Uniform(-30, 30), Uniform(-50, 50)]
#   prior_beta = [Uniform(0.01, 1), Uniform(-1, 1)]
#     Model(
#           alphain = Stochastic(1, () -> prior_alpha),
#           betain = Stochastic(1, () -> prior_beta),
#           alpha = Logical((alphain, nuin) -> polynu(nuin, alpha)),
#           beta = Logical((betain, nuin) -> polynu(nuin, beta)),
#           xout = Stochastic(1, (alpha, beta) -> Gamma(alpha, beta)),
#           )
# end

# function show_model(model::Mamba.Model)
#   using GraphViz
#   display(Graph(graph2dot(model)))
# end

## function log_likelihood(frame::DataFrame, param)
##     @parallel (+) for
## end

# type not strictly needed
## runid::Int64 = 9

# splicing: pass returned tuple into next call as individual arguments
raw_data = read_data()
## packets = filter_positive(raw_data...)
## nus = sub(packets, 1, :)
## energies = sub(packets, 2, :)
## println(size(packets), size(nus), size(energies))

frame = filter_positive(raw_data...)
transform_data!(frame)

llh = logtarget(frame)
θ = [1.52444e+00, 1.85812e-01, -3.33592e+00, 5.49275e+00, 1/7.46542e+01, 0]
θ = [1.52444e+00, 0, 0, 0, 1/7.46542e+01, 0]
θ = [1.5244e+00, 1/7.46542e+01]
println("llh(θ) = $(llh(θ))")

res = optimize(llh, θ, method=BFGS(), ftol=1e-20, grtol=1e-20,
               show_trace=true, autodiff=false)
println(res)

# m = factory(frame[:nu], frame[:x])
# show_model(m)

# Plots works directly with data frames
#histogram(frame, :energies)
## histogram(frame, :energies)
## Plots.pdf("testPlots.pdf")

# Local Variables:
# compile-command:"julia test.jl"
# End:
