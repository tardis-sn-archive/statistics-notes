include("tardis.jl")
using tardis

using DataFrames
using FactCheck
using Polynomials

FactCheck.setstyle(:compact)

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
    transform_data!(frame)
    f, ∇, H = targetfactory(frame, 2, 1)
    println(∇([1.1, 2.2, 3.3]))
end
