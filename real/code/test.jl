include("tardis.jl")
using tardis

using DataFrames
using FactCheck
using Polynomials

FactCheck.setstyle(:compact)

# to be sorted as DataFrame(nus=[0.1, 0.38, 0.5], energies=[1.1, 2.7, 1.2])
mockdata() = DataFrame(nus=[0.1, 0.5, 0.14, 0.38], energies=[1.1, 1.2, -2.3, 2.7])

facts("readdata") do
    nus, energies = readdata(132)
    @fact length(nus) --> length(energies)
    @fact length(nus) --> 100000
end

facts("filter positive") do
    en = [-1.1, 3.3, 23.3, -14, 43.3]
    x = ones(en)
    raw = mockdata()
    frame = filter_positive(raw[:nus], raw[:energies])

    @fact frame[:energies][1] --> greater_than(0)
    @fact length(frame[:energies]) --> 3
    @fact length(frame[:nus]) --> 3
end

facts("update_polynomials!") do
    p1 = Poly([1.1, 2.2, 3.3])
    p2 = Poly([1.1])
    θ = [1., 2, 3, 4]

    update_polynomials!(θ, p1, p2)
    @fact p1[0] --> 1
    @fact p1[1] --> 2
    @fact p1[2] --> 3
    @fact p2[0] --> 4

    @fact polyval(p1, 1.1) --> 1 + 2.2 + 3 * 1.1^2
    @fact polyval(p2, 2) --> 4
end

facts("symmetrize!") do
    A = [1 2; 0 4]
    symmetrize!(A)

    @fact A[1] --> 1
    @fact A[2] --> 2
    @fact A[3] --> 2
    @fact A[4] --> 4
end

facts("allocations") do
    grad, hessian = allocations(3)

    @fact length(grad) --> 3
    @fact length(hessian) --> 9
    @fact size(hessian) --> (3, 3)
end

facts("νpower") do
    @fact νpower(1.1, 2, 3) --> 1.3310000000000004
end

facts("transform_data!") do
    frame = mockdata()
    transform_data!(frame, 2.0)

    @fact frame[:nus][1] --> 0.05
    @fact frame[:nus][2] --> 0.07
    @fact frame[:nus][3] --> 0.19
    @fact frame[:nus][4] --> 0.25
    @fact frame[:energies][1] --> 1 - 1.1 / (2.7*(1+1e-6))
end

facts("factory") do
    data = mockdata()
    frame = filter_positive(data[:nus], data[:energies])
    transform_data!(frame, 2.0)

    f, ∇f!, Hf! = targetfactory(frame, 2, 1)
    θ = [1.1, 2.2, 3.3]

    # evaluate log likelihood manually. We know prior is positive
    p = Poly(θ[1:2])
    α = [polyval(p, νᵢ) for νᵢ in frame[:nus]]
β = [θ[3] for i in 1:3]
    x = frame[:energies]
    ν = frame[:nus]
    # minus for minimization
llh = -sum([loggamma(x[i], α[i], β[i]) for i in 1:3])
    @fact f(θ) --> llh

    ∇res, Hres = allocations(3)

    # evaluate gradient manually
∇_α = -[sum([(log(β[i])-digamma(α[i])+log(x[i])) * ν[i]^(m-1) for i in 1:3]) for m in 1:2]
∇_β = -sum([α[i] / β[i] - x[i] for i in 1:3])
    ∇f!(θ, ∇res)
    @fact ∇res[1:2] --> ∇_α
    @fact ∇res[3] --> ∇_β

    # evaluate Hessian manually: no extra minus sign here compared to paper
    Hf!(θ, Hres)
    # alpha block
    @fact Hres[1,1] --> sum(trigamma, α)
@fact Hres[1,2] --> sum([trigamma(α[i])*ν[i] for i in 1:3])
    @fact Hres[2,1] --> Hres[1,2]
@fact Hres[2,2] --> sum([trigamma(α[i])*ν[i]^2 for i in 1:3])

    # α β block
@fact Hres[1,3] --> -sum([1.0 / β[i] for i in 1:3])
@fact Hres[2,3] --> -sum([ν[i] / β[i] for i in 1:3])

    # β β component
@fact Hres[3,3] --> sum([α[i] / β[i]^2 for i in 1:3])
end
