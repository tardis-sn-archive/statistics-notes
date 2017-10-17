function simps(y::Vector, x::Union{Vector,Range})
    n = length(y)-1
    n % 2 == 0 || error("`y` length (number of intervals) must be odd")
    length(x)-1 == n || error("`x` and `y` length must be equal")
    h = (x[end]-x[1])/n
    # julia 0.5
    # s = sum(slice(y,1:2:n) + 4slice(y,2:2:n) + slice(y,3:2:n+1))
    # julia v0.6
    s = sum(view(y,1:2:n) + 4view(y,2:2:n) + view(y,3:2:n+1))
    return h/3 * s
end

function norm_mean_var(Qs, res, mod=false)
    norm = simps(res, Qs)
    if (mod) res ./= norm end
    μ = simps(res .* Qs, Qs)
    σ2 = simps(res .* (Qs - μ).^2, Qs)
    norm, μ, σ2
end

K = 1000
l1 = zeros(K); l2 = zeros(l1) # plug-in estimators
l1B = zeros(l1); l2B = zeros(l1) # Bayes estimators
pred_mean=zeros(l1); pred_var=zeros(l1) #posterior predictive
for k = 1:K
    n = rand(poisson)
    println("k=$k, n=$n")

    data = rand(dist, n)
    l1[k] = sum(data); l2[k] = n*moment(data, 2, 0);
    logr = sum(log(data))
    a = 0.5 # Jeffreys prior
    # l2B[k] = TardisPaper.Predict.variance_by_cubature(a, n, l1[k], logr; βmax=150)
    l2B[k] = TardisPaper.Predict.variance_by_laplace(a, n, l1[k], logr)
    # res = map(Q->TardisPaper.Predict.by_cubature(Q, a, n, l1[k], logr; reltol=1e-3, ε=1e-3)[1], Qs)
    # pred_mean[k], pred_var[k] = norm_mean_var(Qs, res)[2:end]
end

# estimate the mean
dist = Gamma(1.5, 1/60.)
n = 10
Qs = linspace(0.01, 1, 101)
data = rand(dist, n); q = sum(data); logr = sum(log.(data));
PQ0 = [TardisPaper.Predict.true_by_cubature(Q0, 0.5, n, q, logr; reltol=1e-4) for Q0 in Qs]
Plots.plot(Qs, PQ0; show=true, lab="Q0")
norm_mean_var(Qs, PQ0)

# variance scales roughly with n as expected from Poisson uncertainty
# n 10    100    500
# V 0.011 0.0947 0.446
