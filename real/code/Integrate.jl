"""use Cubature or Laplace"""
module Integrate
using Cubature, Optim, PyCall

"""
            by_cubature(logf, xmin, xmax;<keyword arguments>)

        Integrate the 1D function `f` from `xmin` to `xmax` by quadrature. The
        upper limit `xmax` may be ∞ but `xmin` must be finite.  Any keyword
        arguments are passed on to `Cubature.hquadrature`.

        # Examples
        ```jldoctest
        julia> by_cubature()
        ```
        """
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
        h = t -> (counter += 1; exp(logf(t)))
    else
        # transform variables, mutate x in place
        x = Vector(xmax)
        trafo = t -> begin
            x .= t
            jacobian = 1.0
            for i in indices
                x[i] = xmin[i] + t[i]/(1-t[i])
                jacobian *= 1/(1-t[i])^2
            end
            return jacobian
        end
        h = t -> begin
            counter += 1
            jacobian = trafo(t)
            exp(logf(x)) * jacobian
        end
    end
    estimate, σ = hcubature(h, a, b; kwargs...)
    return estimate, σ, counter
end

"Use Simpson's rule to evalue the integral of a function with `values`
evaluated at `points`, and compute the mean and standard deviation."
function simpson(points, values)
    @pyimport scipy.integrate as si
    normalization = si.simps(values, points)
    μ = si.simps(values .* points, points)
    σ2 = si.simps(values .* (points - μ).^2, points)
    normalization, μ, sqrt(σ2)
end

"Laplace approximation to the log(integral) over f using the Hessian
at the mode, -Hf(θ). Both f and Hf are on the log scale."
laplace(logf::Real, log_det_H::Real, dim::Integer) = logf + dim/2 * log(2pi) - 1/2*log_det_H
laplace(logf::Real, H::Matrix) = laplace(logf, log(det(H)), size(H)[1])

"""
Integrate f by Laplace.

Use Newton's method with autodiff. Return the optimization result and
the integral estimate. The Hessian is stored in H. If it passed in, it
is modified """

function by_laplace!(logf::Function, xinit::Vector, H = zeros(length(xinit), length(xinit)))
    # negate minimization
    target = x -> -logf(x)
    res = optimize(target, xinit, Newton(), OptimizationOptions(autodiff=true))
    ForwardDiff.hessian!(H, target, Optim.minimizer(res))
    res, laplace(-Optim.minimum(res), H)
end



end # Integrate
