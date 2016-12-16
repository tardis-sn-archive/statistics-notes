import Distributions, PyCall, TardisStats
using FredPlots

α0 = 1.5; β0 = 60.; λ0 = 500
νmin=0.
νmax=10.

# define order of linear models
αOrder = 1
βOrder = 1
npackets=1000

# read in data from tardis and create posterior
dim = αOrder + βOrder
∇res, Hres = TardisStats.optim.allocations(dim)
dataframe, (P, ∇P!, HP!) = TardisStats.optim.problem(αOrder=αOrder, βOrder=βOrder, run=99, npackets=npackets)

# now change the energies
PyCall.@pyimport numpy.random as nprand
nprand.seed(1358)
PyCall.@pyimport scipy.stats as stats
dist0 = stats.gamma(α0, scale=1/β0)
# crucial: change dataframe in place, else optimization runs with old energies!
dataframe[:energies][:] = dist0[:rvs](size=length(dataframe[:energies]))
# Plots.histogram(dataframe[:energies])

# the observed number of packets in the bin
nmin=searchsortedfirst(dataframe[:nus], νmin)
nmax=searchsortedlast(dataframe[:nus], νmax)
n = nmax - nmin + 1
x = sum(dataframe[:energies][nmin:nmax])
n,x

@time maxP, posterior_mode, ret = TardisStats.optim.run_nlopt(dataframe, P, ∇P!, HP!,
    αOrder, βOrder, xtol_rel=1e-4, init=1.001*[α0, β0])
HP!(posterior_mode, Hres)
evidence = TardisStats.optim.laplace(maxP, Hres)
println("got $maxP at $(posterior_mode) (returned $ret) and evidence $evidence")
