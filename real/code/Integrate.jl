"""use Cubature or Laplace"""
module Integrate
using Cubature

function by_cubature(logf::Function, xmin::Real, xmax::Real; kwargs...)
    # count function evaluations
    counter = 0
    h = t -> (counter += 1; exp(logf(t)))

    # transform variables
    if xmax === Inf
        h = t -> (counter += 1; exp(logf(xmin + t/(1-t))) * 1/(1-t)^2)
        xmin, xmax = 0, 1
    end
    estimate, σ = hquadrature(h, xmin, xmax; kwargs...)
    return estimate, σ, counter
end

function by_cubature(logf::Function, xmin::Vector, xmax::Vector; kwargs...)
    # count function evaluations
    counter = 0

    # keep independent limits in case of transform
    a = copy(xmin)
    b = copy(xmax)

    # count parameters to transform and their index in the arg vector
    ninf = sum(x -> x == Inf, xmax)
    indices = Vector{Int16}(ninf)
    offset = 1
    for (i, xi) in enumerate(xmax)
        if xi == Inf
            indices[offset] = i
            offset += 1
            a[i] = 0
            b[i] = 1
        end
    end

    if ninf == 0
        h = function(t) counter += 1; exp(logf(t)) end
    else
        # transform variables, mutate x in place
        x = Vector(xmax)
        trafo = function(t)
            x .= t
            jacobian = 1.0
            for i in indices
                x[i] = xmin[i] + t[i]/(1-t[i])
                jacobian *= 1/(1-t[i])^2
            end
            return jacobian
        end
        h = function(t)
            counter += 1
            jacobian = trafo(t)
            exp(logf(x)) * jacobian
        end
    end
    estimate, σ = hcubature(h, a, b; kwargs...)
    return estimate, σ, counter
end

end # Integrate
