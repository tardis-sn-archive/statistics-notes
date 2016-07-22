using DataFrames
using HDF5
using Optim
using Polynomials

# read in data
# function readdata(run, maxelements=false)
#     # todo use maxelements
#     file = h5open("../../posterior/real_tardis_250.h5", "r")
#     en = file["energies"][:, run]
#     nus = file["nus"][:, run]
#     close(file)
#     return nus, en
# end

"""
read energies and frequencies from a particular run
"""

function readdata(runid=10, filename="../../posterior/real_tardis_250.h5")
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
end

"""

Sort by frequency, then rescale and flip the energies, then rescale
the frequencies.

"""

function transform_data!(frame::DataFrame, nuscale::Float64=1e15)
    sort!(frame, cols=:nus)

    # update energies

    # reference to the frame if we index directly
    # things like en /= 2 operate on a copy of the data
    en = frame[:energies]
    maxen = (1.0 + 1e-6) * maximum(en)
    for i in eachindex(en)
        en[i] = 1.0 - en[i] / maxen
    end

    frame[:nus] /= nuscale

    println("min nu = $(frame[:nus][1]), max en = $(maximum(en))")

    # return nothing, just mutate input
    nothing
end

function loggamma(x::Float64, α::Float64, β::Float64)
  -α * log(β) - lgamma(α) + (α-1)*log(x) - x / β
end

"""

Create all target functions, return triples of log(f), grad(f),
Hessian(f).

"""

function targetfactory(frame::DataFrame)
    αOrder::Int32 = 1
    βOrder::Int32 = 1
    αPoly = Poly(Vector{typeof(θ[1])}(αOrder))
    βPoly = Poly(Vector{typeof(θ[1])}(βOrder))
    ν::Vector{Float64} = frame[:nus]
    x::Vector{Float64} = frame[:energies]

    # closure: needs frame::Dataframe in scope
    log_likelihood = function (θ::Vector)
        res::Float64 = 0
        α::Float64 = 0
        β::Float64 = 0
        invalid = -1e100 # oftype(1.0, -Inf)
        for n in 1:αOrder
            @inbounds αPoly.a[n] = θ[n]
        end
        for n in 1:βOrder
            @inbounds βPoly.a[n] = θ[αOrder + n]
        end

        for i in eachindex(ν)
            @inbounds α = polyval(αPoly, ν[i])
            @inbounds β = polyval(βPoly, ν[i])

            # prior: reject point if α or β too small
            if α < 1.0
                return invalid
            end

            if β < 0.0
                return invalid
            end

            res += loggamma(x[i], α, β)

            if !isfinite(res)
                return invalid
            end
        end
        # suitable for minimization
       -res
    end # log_likelihood
end


run = 11
nus, en = readdata(run)
frame = filter_positive(nus, en)
transform_data!(frame)
