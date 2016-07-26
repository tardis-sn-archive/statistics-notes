module tardis

using DataFrames
using HDF5
using Optim
using Polynomials

export allocations, filter_positive, loggamma, νpower, readdata, symmetrize!, targetfactory, transform_data!, update_polynomials!

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

    # println("min nu = $(frame[:nus][1]), max  max en = $(maximum(en))")

    # return nothing, just mutate input
    nothing
end

function loggamma(x::Float64, α::Float64, β::Float64)
    -α * log(β) - lgamma(α) + (α-1)*log(x) - x / β
end

"""
Reset the coefficients of the polynomials to θ
"""
function update_polynomials!(θ::Vector, αPoly, βPoly)
    N::Int64 = length(αPoly)
    # aPoly[0]= aPoly.a[1] is the constant
    for n in 1:N
        @inbounds αPoly.a[n] = θ[n]
    end
    for n in 1:length(βPoly)
        @inbounds βPoly.a[n] = θ[N + n]
    end
end

"""
Copy upper triangular part of matrix into lower triangle. Do not touch the diagonal.

"""
function symmetrize!(H::Array)
    for m in 1:size(H)[1]
        for n in m+1:size(H)[2]
            H[n, m] = H[m, n]
        end
    end
end

function allocations(dim::Int64)
    return zeros(dim), zeros((dim, dim))
end

"""
Raise ν to the power of m + n, where m and n are 1-based indices.
"""
νpower(ν, m, n) = ν^(m + n - 2)

@enum STATE kLikelihood kMean

"""
Create all target functions, return triples of log(f), grad(f),
Hessian(f).
"""
function targetfactory(frame::DataFrame, αOrder::Int64, βOrder::Int64;
                       X=Nullable{Float64}, N=Nullable{Int64}, νbin=Nullable{Float64})
    # references
    ν::Vector{Float64} = frame[:nus]
    x::Vector{Float64} = frame[:energies]
    Ψ = digamma
    Ψ′ = trigamma

    # possible input states
    local state::STATE
    if X == Nullable{Float64} && N == Nullable{Int64} && νbin == Nullable{Float64}
        state = kLikelihood
    elseif X > 0 && N >= 0 && νbin >= 0
        state = kMean
    else
        error("Invalid input values: X = $X, N = $N, ν = $νbin")
    end

    # allocations
    αPoly = Poly(ones(αOrder))
    βPoly = Poly(ones(βOrder))

    # closures need frame::Dataframe in scope!

    #
    # log_likelihood
    #
    log_likelihood = function(θ::Vector)
        update_polynomials!(θ, αPoly, βPoly)

        res::Float64 = 0
        α::Float64 = 0
        β::Float64 = 0
        invalid = 1e100 # oftype(1.0, -Inf)

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
        return -res
    end # log_likelihood

    ∇log_likelihood! = function(θ::Vector, ∇::Vector)
        update_polynomials!(θ, αPoly, βPoly)
        ∇[:] = 0.0

        for i in eachindex(ν)
            @inbounds α = polyval(αPoly, ν[i])
            @inbounds β = polyval(βPoly, ν[i])

            tmp = log(β) - Ψ(α) + log(x[i])
            for n in 1:length(αPoly)
                ∇[n] += tmp
                tmp *= ν[i]
            end

            tmp = α / β - x[i]
            for n in 1:length(βPoly)
                ∇[length(αPoly) + n] += tmp
                tmp *= ν[i]
            end
        end
        # minus for minimization
        ∇[:] *= -1.0
    end

    log_likelihood_hessian! = function(θ::Vector, H::Matrix)
        update_polynomials!(θ, αPoly, βPoly)
        H[:] = 0.0

        for i in eachindex(ν)
            @inbounds α = polyval(αPoly, ν[i])
            @inbounds β = polyval(βPoly, ν[i])

            # directly modify only upper triangular part and diagonal

            # α vs α block
            tmp = Ψ′(α)
            for m in 1:length(αPoly)
                for n in m:length(αPoly)
                    H[m, n] += tmp * νpower(ν[i], m, n)
                end
            end
            # α vs β block
            tmp = 1.0 / β
            offset = length(αPoly)
            for m in 1:length(αPoly)
                for n in 1:length(βPoly)
                    H[m, offset + n] -= tmp * νpower(ν[i], m, n)
                end
            end
            # β vs β block
            tmp = α / β^2
            for m in 1:length(βPoly)
                for n in m:length(βPoly)
                    H[offset + m, offset + n] += tmp * νpower(ν[i], m, n)
                end
            end
        end
        symmetrize!(H)

        # no minus sign, expressions in paper already have the minus
        return H
    end

    #
    # mean
    #
    posterior_mean = function(θ::Vector)
        update_polynomials!(θ, αPoly, βPoly)
        @inbounds α = polyval(αPoly, νbin)
        @inbounds β = polyval(βPoly, νbin)
        return log_likelihood(θ) + loggamma(X, N * α, β)
    end

    ∇posterior_mean = function(θ::Vector)
        update_polynomials!(θ, αPoly, βPoly)
        @inbounds α = polyval(αPoly, νbin)
        @inbounds β = polyval(βPoly, νbin)

        # mutates ∇
        ∇log_likelihood!(θ)

        # α
        tmp = N * (log(β) - Ψ(N * α) + log(X))
        for i in 1:length(αPoly)
            ∇[i] += tmp
            tmp *= νbin
        end

        # β
        tmp = N * α / β - X
        offset = length(αPoly)
        for i in 1:length(βPoly)
            ∇[offset + i] += tmp
            tmp *= νbin
        end

        return ∇
    end

    posterior_mean_hessian = function(θ::Vector)
        update_polynomials!(θ, αPoly, βPoly)
        @inbounds α = polyval(αPoly, νbin)
        @inbounds β = polyval(βPoly, νbin)

        # directly modify only upper triangular part and diagonal

        # α vs α block
        tmp = N^2 * Ψ′(N * α)
        for m in 1:length(αPoly)
            for n in m:length(αPoly)
                H[m, n] += tmp * νpower(νbin, m, n)
            end
        end
        # α vs β block
        tmp = N / β
        offset = length(αPoly)
        for m in 1:length(αPoly)
            for n in 1:length(βPoly)
                H[m, offset + n] -= tmp * νpower(νbin, m, n)
            end
        end
        # β vs β block
        tmp = N * α / β^2
        for m in 1:length(βPoly)
            for n in m:length(βPoly)
                H[offset + m, offset + n] = tmp * νpower(νbin, m, n)
            end
        end

        symmetrize!(H)
        return H
    end

    # what function triple to return
    if state == kLikelihood
        return log_likelihood, ∇log_likelihood!, log_likelihood_hessian!
    elseif state == kMean
        return posterior_mean, ∇posterior_mean, posterior_mean_hessian
    else
        error("invalid state")
    end
end

end # module
