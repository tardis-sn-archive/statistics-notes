include("tardis.jl")
using tardis
using Optim

run = 11
raw_data = readdata(run)
frame = filter_positive(raw_data...)
transform_data!(frame)
αOrder = 4
βOrder = 2
f, ∇f!, Hf! = targetfactory(frame, αOrder, βOrder)

# initial value
θ = [1.3, 0, 0, 0, 54, 1.0]

res = optimize(
#               f, θ, NelderMead()
               ## f, ∇f!, θ, BFGS(),
               f, ∇f!, Hf!, θ, Newton(),
               OptimizationOptions(iterations=10000))

println(res)
θ = Optim.minimizer(res)
println("f($(θ)) = $(f(θ))")
