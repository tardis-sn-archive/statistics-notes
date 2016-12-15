using Distributions, Plots, PyCall, StatPlots
using FredPlots: setup_pyplot; setup_pyplot()

λ = 500
α = 1.5
β = 60

# dist = Gamma(λ*α, β)
# plot(dist)

@pyimport scipy.stats as stats
dist = stats.gamma(λ*α, scale=1/β)
samples = dist[:rvs](size=10000)
histogram(samples, normed=true)
plot!(Normal(λ*α/β, sqrt(λ*α/β^2*(1+α))))
plot!(Normal(λ*α/β, sqrt(λ*α/β^2)), label="Harr")

poisson = stats.poisson(λ)
nmin = convert(Int64, floor(λ-4*√(λ)))
nmax = convert(Int64, floor(λ+4*√(λ)))
n = collect(nmin:nmax)
y = poisson[:pmf](n)
plot(n, y)

# discrete convolution
Qs = linspace(6, 11, 100)
function sumup(Q)
  res = 0.0
  for N in n
    d = stats.gamma(N*α, scale=1/β)
    res += y[N-nmin+1] * d[:pdf](Q)
  end
  res
end

posterior = Float64[sumup(Q) for Q in Qs]

plot(Qs, posterior, label="sum")
plot!(Normal(λ*α/β, sqrt(λ*α/β^2*(1+α))), label="normal approx.")

# Plot the posterior on \mu, \sigma^2
function make_log_posterior(n, μ_n, a_n, b_n)
    log_posterior(μ, σSq) = -1/2*(log(2π*σSq / n) -(μ - μ_n)^2 / σSq * n) +
        (a_n * log(b_n) - lgamma(a_n) - (a_n + 1) * log(σSq) - b_n / σSq)
end

f = make_posterior(λ, α/β, λ/2, λ/2 * α/β^2)
