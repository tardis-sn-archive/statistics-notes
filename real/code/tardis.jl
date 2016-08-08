module tardis

using DataFrames
using HDF5
using Optim
using Polynomials

export allocations, filter_positive, laplace, loggamma, lognegativebinomial, NegBinom, νpower, PosteriorMean, readdata, symmetrize!, targetfactory, transform_data!, update_polynomials!

"""
read energies and frequencies from a particular run. Limit to read in at most npackets
"""

function readdata(runid=10; filename="../../posterior/real_tardis_250.h5", npackets=typemax(Int64))
    h5open(filename, "r") do file
        npackets = min(size(file["nus"])[1], npackets)
        # read only a single run
        file["nus"][1:npackets,runid], file["energies"][1:npackets,runid]
    end
end

"""
filter_positive(x, y)

For two arrays of same length, filter out elements with negative x and
return a data frame with columns (nu, energies)

"""

function filter_positive(nus, energies)
    @assert length(energies) == length(nus)
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
    α * log(β) - lgamma(α) + (α-1)*log(x) - β * x
end

function lognegativebinomial(N::Real, n::Integer, a::Real)
    tmp = N + n -a + 1
    lgamma(tmp) - lgamma(N+1) - lgamma(n-a+1) - tmp * log(2)
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
Overwrite the lower triangle.

"""
function symmetrize!(H::Array)
    for m in 1:size(H)[1]
        for n in m+1:size(H)[2]
            H[n, m] = H[m, n]
        end
    end
end

function allocations(dim::Integer)
    return zeros(dim), zeros((dim, dim))
end

"""
Raise ν to the power of m + n, where m and n are 1-based indices.
"""
νpower(ν, m, n) = ν^(m + n - 2)

@enum STATE kLikelihood kMean kAll

type PosteriorMean
    ν::Real
    X::Real
    N::Real
end

type NegBinom
    n::Integer
    a::Real
end

"""
Create all target functions, return triples of log(f), grad(f),
Hessian(f). Hessian is -∂²log(f), note the minus sign.
"""
function targetfactory(frame::DataFrame, αOrder::Int64, βOrder::Int64;
                       pm=nothing, evidence=nothing, nb=nothing)
    # references
    ν::Vector{Float64} = frame[:nus]
    x::Vector{Float64} = frame[:energies]
    Ψ = digamma
    Ψ′ = trigamma

    # possible input states
    local state::STATE
    if pm === nothing
        state = kLikelihood
    else
        if pm.X > 0 && pm.N >= 0 && pm.ν >= 0
            state = kMean
        else
            error("Invalid input values: X = $(pm.X), N = $(pm.N), ν = $(pm.ν)")
        end

        if nb !== nothing
            if (nb.n < 0) error("invalid n < 0: $(nb.n)") end
            if (nb.a < 0) error("invalid a < 0: $(nb.a)") end
            # no error, we are fine
            state = kAll
        end
    end

    # allocations
    αPoly = Poly(ones(αOrder))
    βPoly = Poly(ones(βOrder))

    # rescale with evidence
    rescale = 0.0
    if evidence !== nothing
        rescale = evidence # / length(ν)
    end

    # closures need frame::Dataframe in scope!

    #
    # log_likelihood
    #
    log_likelihood = function(θ::Vector)
        update_polynomials!(θ, αPoly, βPoly)

        res::Float64 = 0
        α::Float64 = 0
        β::Float64 = 0

        for i in eachindex(ν)
            @inbounds α = polyval(αPoly, ν[i])
            @inbounds β = polyval(βPoly, ν[i])

            res += loggamma(x[i], α, β)
        end
        return res - rescale
    end # log_likelihood

    ∇log_likelihood! = function(θ::Vector, ∇::Vector)
        update_polynomials!(θ, αPoly, βPoly)
        ∇[:] = 0.0
        α::Float64 = 0
        β::Float64 = 0

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
    end

    log_likelihood_hessian! = function(θ::Vector, H::Matrix)
        update_polynomials!(θ, αPoly, βPoly)
        H[:] = 0.0
        α::Float64 = 0
        β::Float64 = 0

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
    end

    #
    # mean
    #
        posterior_mean = function(θ::Vector)
        update_polynomials!(θ, αPoly, βPoly)
        @inbounds α = polyval(αPoly, pm.ν)
        @inbounds β = polyval(βPoly, pm.ν)
        # N could be a constant or a fit parameter
        const N = (state === kAll) ? θ[end] : pm.N

        return log_likelihood(θ) + loggamma(pm.X, N * α, β)
    end

    ∇posterior_mean! = function(θ::Vector, ∇::Vector)
        # only update log_likelihood
        ∇log_likelihood!(θ, ∇)

        update_polynomials!(θ, αPoly, βPoly)
        @inbounds α = polyval(αPoly, pm.ν)
        @inbounds β = polyval(βPoly, pm.ν)

        # N could be a constant or a fit parameter
        const N = (state === kAll) ? θ[end] : pm.N

        # α
        tmp = (log(β * pm.X) - Ψ(N * α)) * N

        for i in 1:length(αPoly)
            ∇[i] += tmp
            tmp *= pm.ν
        end

        # β
        tmp = N * α / β - pm.X

        offset = length(αPoly)
        for i in 1:length(βPoly)
            ∇[offset + i] += tmp
            tmp *= pm.ν
        end

        if state === kAll
            ∇[end] += (log(β * pm.X) - Ψ(N * α)) * α
            println(∇[end])
        end
    end

    posterior_mean_hessian! = function(θ::Vector, H::Matrix)
        log_likelihood_hessian!(θ, H)

        update_polynomials!(θ, αPoly, βPoly)
        @inbounds α = polyval(αPoly, pm.ν)
        @inbounds β = polyval(βPoly, pm.ν)

        # N could be a constant or a fit parameter
        N = (state === kAll) ? θ[end] : pm.N

        # directly modify only upper triangular part and diagonal
        # α vs α block
        tmp = N^2 * Ψ′(N * α)
        for m in 1:length(αPoly)
            for n in m:length(αPoly)
                H[m, n] += tmp * νpower(pm.ν, m, n)
            end
        end
        # α vs β block
        tmp = N / β
        offset = length(αPoly)
        for m in 1:length(αPoly)
            for n in 1:length(βPoly)
                H[m, offset + n] -= tmp * νpower(pm.ν, m, n)
            end
        end
        # β vs β block
        tmp = N * α / β^2
        for m in 1:length(βPoly)
            for n in m:length(βPoly)
                H[offset + m, offset + n] += tmp * νpower(pm.ν, m, n)
            end
        end

        if state === kAll
            # α vs N
            tmp = -log(β) + Ψ(N * α) + N * α * Ψ'(N * α) - log(pm.X)
            for m in 1:length(αPoly)
                # β power = 1 => no effect
                H[m, end] += tmp * νpower(pm.ν, m, 1)
            end

            # β vs N
            tmp = -α / β
            for m in 1:length(βPoly)
                # α power = 1 => no effect
                H[offset + m, end] += tmp * νpower(pm.ν, 1, m)
            end

            # N vs N
            H[end, end] += α^2 * Ψ'(N * α)
        end

        symmetrize!(H)
    end

        all_mean = function(θ::Vector)
            return posterior_mean(θ) + lognegativebinomial(θ[end], nb.n, nb.a)
        end

        ∇all_mean! = function(θ::Vector, ∇::Vector)
            ∇posterior_mean!(θ, ∇)
            N = θ[end]
            println("N $N, n $(nb.n), a $(nb.a)")
            ∇[end] += Ψ(N + nb.n - nb.a + 1) - Ψ(N+1) - log(2)
        end

            all_mean_hessian! = function(θ::Vector, H::Matrix)
                posterior_mean_hessian!(θ, H)
                @inbounds α = polyval(αPoly, pm.ν)
                N = θ[end]
                H[end, end] += Ψ'(N+1) - Ψ'(N+nb.n-nb.a+1)
            end


        # what function triple to return
        if state === kLikelihood
            return log_likelihood, ∇log_likelihood!, log_likelihood_hessian!
        elseif state === kMean
            return posterior_mean, ∇posterior_mean!, posterior_mean_hessian!
        elseif state === kAll
            return all_mean, ∇all_mean!, all_mean_hessian!
        else
            error("invalid state: $state")
        end
    end

        """

Laplace approximation to the log(integral) over f using the Hessian at
the mode, -Hf(θ). Both f and Hf are on the log scale.

"""
        laplace(logf::Real, log_det_H::Real, dim::Integer) = logf + dim/2 * log(2pi) - 1/2*log_det_H
        laplace(logf::Real, H::Matrix) = laplace(logf, log(det(H)), size(H)[1])

    end # module
