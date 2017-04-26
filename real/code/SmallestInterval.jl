module SmallestInterval

using Distributions, Interpolations

"Find critical level for smallest interval containing α probability
mass for the array `P` describing the density at a bin center and bins
regularly spread out."

function level(P::AbstractArray, α::AbstractFloat=0.6827)
    (0 ≤ α ≤ 1) || error("Invalid α=$α")

    # remember indices
    perm = sortperm(P, rev=true)
    sortP = sort(P, rev=true)
    norm = sum(P)
    sortP ./= norm
    cumP = cumsum(sortP)
    # overcovers: index of first element beyond α
    i = searchsortedfirst(cumP, α)

    # original index into P
    perm[i]
end

"`k::Integer` is the Gaussian sigma passed to `level`"
level(P::AbstractArray, k::Integer=1) = level(P, 2cdf(Normal(0,1), k)-1)

"""Find index of a simply connected smallest interval containing `α` probability.

Assumes a unimodal distribution to find interval around mode.
"""
function connected(P::AbstractArray, α::Real=0.6827)
    # critical index
    i = level(P, α)

    ###
    # assume unimodal distribution, is `i` at left or right edge of interval?
    ###
    imax = indmax(P)

    if imax < i
        # i at right edge
        return findfirst(x -> x≥P[i], P[1:imax]):i
    else
        # i at left edge, add imax as index offset because search starts at mode
        iright = findlast(x -> x≥P[i], P[imax:end])
        return i:imax+iright-1
    end
end

function upscale(Qs::Range, P::AbstractArray, N::Integer=500)
    # why scale? Interpolations.jl assumes unit spacing between elements of P
    itp_cubic = scale(interpolate(P, BSpline(Cubic(Line())), OnGrid()), Qs)
    fineQs = linspace(Qs[1], Qs[end], N)
    fineQs, map(Q->itp_cubic[Q], fineQs);
end

function upscale(Qs::AbstractArray, P::AbstractArray, N::Integer=500)
    warn("upscale: assuming regularly spaced input")
    upscale(linspace(Qs[1], Qs[end], length(Qs)), P, N)
end

end # module
