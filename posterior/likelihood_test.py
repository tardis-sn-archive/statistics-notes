from __future__ import print_function

from likelihood import *
from mixture import *

from matplotlib import pyplot as plt
import numpy as np
import pytest
import scipy

def test_log_pochhammer():
    args = ((13, 11), (11, 13), (1, 2))
    res  = (31.6195, 36.5023, 0.693147)
    for a, r in zip(args, res):
        np.testing.assert_almost_equal(log_pochhammer(*a), r, decimal=3)


def test_poisson_posterior_predictive():
    args = ((13, 11), (11, 13), (1, 2), (12, 12))
    res  = np.exp([-2.63578, -2.554, -1.50972, -2.513172799864464])
    for a, r in zip(args, res):
        np.testing.assert_almost_equal(poisson_posterior_predictive(*a), r, decimal=3)


def test_binary_search():
    data = (1, 3, 6, 7, 9, 9, 9, 9, 10)
    keys = (0.1, 1, 2.2, 7, 9, 9.2, 11)
    imin = (0,   0, 1,   3, 4, 8)
    imax =      (0, 0,   3, 7, 7,   8)

    for k, i in zip(keys[:-1], imin):
        assert binary_search_min(data, k) == i

    # key larger than all elements in ``data``
    with pytest.raises(KeyError):
        binary_search_min(data, keys[-1])

    for k, i in zip(keys[1:], imax):
        assert binary_search_max(data, k) == i

    # key larger than all elements in ``data``
    with pytest.raises(KeyError):
        binary_search_max(data, keys[0])


def test_expand_bin():
    N = 1000
    K = 3000

    # easy: no boundary hit
    imin, imax = 927, 1433
    missing = N - (imax - imin)
    assert expand_bin(N, imin, imax, K) == (680, 1680)

    # hit lower boundary
    imin, imax = 227, 733
    missing = N - (imax - imin)
    assert expand_bin(N, imin, imax, K) == (0, 1000)

    # hit upper boundary
    imin, imax = K - 733, K - 227
    missing = N - (imax - imin)
    assert expand_bin(N, imin, imax, K) == (2000, 3000)



@pytest.fixture(scope="session")
def hdf5():
    import h5py
    f = h5py.File("real_tardis_250.h5", 'r')
    return f


@pytest.fixture(scope="session")
def single_run(hdf5):
    # read in a single run
    run = 7
    energies = hdf5["/energies"][run]
    nus = hdf5["/nus"][run]

    assert len(energies) == len(nus)

    # filter out negative energies and tack arrays together
    filter = energies > 0
    arr = np.core.records.fromarrays([nus[filter], energies[filter]],
                                     names='nus,energies')

    # sort in place by frequency
    arr.sort(order='nus')

    return arr


# @pytest.mark.skipif(True, reason="Too many plots")
def test_prob_L_given_N(single_run):
    imin, imax =  binary_search_min_max(single_run.nus, 1.199e15, 1.2e15)
    N = imax - imin
    nsum_samples = 2000
    jmultimin, jmultimax = expand_bin(nsum_samples * N, imin, imax, len(single_run))

    print('%d packets in bin, request %d summed samples, got %d' %
          (N, nsum_samples, (jmultimax - jmultimin) // N))

    in_bin = single_run[imin:imax]
    lmultibin = single_run.energies[jmultimin:jmultimax]

    print("Total lum %g" % in_bin.energies.sum())
    # plt.hist(in_bin.energies)
    plt.hist(lmultibin, bins=80)
    # plt.hist(single_run.energies[-1000:], bins=100)
    plt.savefig("energies_all.pdf")
    plt.clf()


    # force empirical distribution
    kde = prob_L_given_N_empirical(N, lmultibin)

    # For N large, the CLT says the mode should be near the Gaussian approx.
    # Compare to mean, sigma taken from bin alone
    # => see 1/N bias in mean estimation
    mean = N * np.mean(in_bin.energies)
    sigma = math.sqrt(N * np.var(in_bin.energies))

    l = np.linspace(mean - 10 * sigma, mean + 10 * sigma, 300)
    plt.plot(l, kde(l))
    plt.plot(l, scipy.stats.norm(loc=mean, scale=sigma).pdf(l))

    # rely on default to get Gaussian approximation
    gaussian = prob_L_given_N(N, lmultibin, np.mean(lmultibin), np.std(lmultibin))
    plt.plot(l, gaussian(l))

    plt.axvline(in_bin.energies.sum(), color='black')

    plt.savefig("energies_empirical.pdf")


@pytest.mark.skipif(True, reason="Not ready yet")
def test_prob_L_given_N_CLT():
    # compare to my mathematica test
    np.testing.assert_allclose(
        prob_L_given_N(12, None, mean=1.4022297856077416e+38,
        variance=1.2255043910166591e+73, N_critical=1)(1.66565e+39),
        9.9483e-40)


@pytest.mark.skipif(True, reason="Not ready yet")
def test_prob_L_given_theta(single_run):
    print((len(single_run), single_run[0]))
    imin, imax =  binary_search_min_max(single_run.nus, 1.1998e15, 1.2e15)
    N = imax - imin
    nsum_samples = 250
    jmultimin, jmultimax = expand_bin(nsum_samples * N, imin, imax, len(single_run))
    print((imin, imax))
    print((single_run[imin], single_run[imax]))
    print((jmultimin, jmultimax))
    print((np.mean(single_run.energies[jmultimin:jmultimax]),
           np.var(single_run.energies[jmultimin:jmultimax])))
    print(poisson_posterior_predictive(N, N))

    # check for value of L that matches what was observed
    L = single_run.energies[imin:imax].sum()
    print('%d packets in bin with total luminosity %g, request %d summed samples, got %d' %
          (N, L, nsum_samples, (jmultimax - jmultimin) // N))

    prob_L_given_theta(L, single_run.energies, imin, imax, eps=1e-5)


def test_interpolator(single_run):
    samples = single_run.energies[5000:5005]
    scale = 1e38
    samples /= scale

    print('mean', samples.mean(), 'variance', samples.var())
    print('skew', scipy.stats.skew(samples), 'kurtosis', scipy.stats.kurtosis(samples, fisher=False))

    x = np.linspace(0.995 * samples.min(), 1.005 * samples.max(), 100)

    plt.clf()
    plt.hist(samples, bins=50, histtype='stepfilled',
             normed=True, color='grey', alpha=0.3)

    # need to adjust prior if samples are rescaled by 1e38 because covariance prior too wide (?)
    # p = PypmcInterpolator(samples, k=3, iterations=500, verbose=True)
    # y = np.exp(p(x))
    # print(p.mixture.components[0].sigma)
    # plt.plot(x, y)

    optim = alpha_mu_max_likelihood(samples)
    print(optim)
    # a, alpha, mu, rhat = optim.x
    # y = [alpha_mu(xi, alpha, mu, rhat, a) for xi in x]
    # plt.plot(x, y)

    plt.savefig("mixture_test2.pdf")


def test_max_likelihood(single_run):
    i = 13110
    samples = single_run.energies[i:i+1500]
    # normalized for easier numerics
    samples *= 1e-38

    print('length', len(single_run.energies))
    mean, var = samples.mean(), samples.var()
    print('mean', mean, 'variance', var)
    print('skew', scipy.stats.skew(samples), 'kurtosis', scipy.stats.kurtosis(samples, fisher=False))

    loc = 1.5 * samples.max()
    optim = scipy.stats.gamma.fit(loc - samples)
    a = loc - optim[1]
    print()
    print("a", a, "theta", optim[2], "alpha", optim[0])

    print("Same with packaged method", gamma_max_likelihood(samples))

    x = np.linspace(samples.min(), a, 800)
    y = scipy.stats.gamma.pdf(a - x, optim[0], scale=optim[2])
    plt.clf()
    import astroML.plotting
    hist_kw = dict(bins=30, histtype='stepfilled', normed=True, alpha=0.5, color='grey')
    astroML.plotting.hist(samples, **hist_kw)
    plt.plot(x, y)
    plt.scatter(samples, np.zeros_like(samples), marker='+', color='red')
    plt.ylim(-0.5, None)
    plt.xlim(0.99 * x[0], 1.01 * x[-1])
    plt.title("%d samples, N=1" % len(samples))
    plt.savefig("max_likelihood.pdf")

    # now the sum of n samples
    n = 3
    alpha = n * optim[0]
    sum_samples = np.array([samples[i:i + n].sum() for i in range(len(samples) // n)])
    x = np.linspace(0.999*sum_samples.min(), 1.001*sum_samples.max(), 300)
    y = scipy.stats.gamma.pdf(n * a - x, alpha, scale=optim[2])
    # z1 = scipy.stats.norm.pdf(n * a - x, loc=0, scale=optim[2] * np.sqrt(alpha))
    # CLT
    z = scipy.stats.norm.pdf(x, loc=n * mean, scale=np.sqrt(n * var))

    plt.clf()
    hist_kw['bins'] = 'blocks'
    astroML.plotting.hist(sum_samples, label='hist', **hist_kw)
    plt.plot(x, y, label='Gamma')
    plt.plot(x, z, label='Gauss')
    plt.title("%d samples, N=%d" % (len(sum_samples), n))
    plt.legend(loc='upper left')
    plt.savefig("max_likelihood_sum.pdf")


def test_amoroso_binned_max_likelihood(single_run):

    samples = single_run.energies[5000:8000]
    scale = 1e38
    samples /= scale

    nbins = 80
    optim = amoroso_binned_max_log_likelihood(samples, nbins=nbins)
    optim[3] = 1.0
    print(optim)
    print("max value", samples.max())

    plt.figure()
    plt.hist(samples, bins=nbins, histtype='stepfilled',
             normed=True, color='grey', alpha=0.3)
    x = np.linspace(0.995 * samples.min(), 1.005 * samples.max(), 100)
    y = amoroso_pdf(x, optim)
    plt.plot(x, y)
    plt.savefig("amoroso_amoroso_binned_max_log_likelihood.pdf")


def test_alpha_mu():
    # compare to mathematica output
    np.testing.assert_approx_equal(alpha_mu(1.2, 1.1, 1.4, 0.4, 0.2), 0.175734, 5)
    np.testing.assert_approx_equal(alpha_mu(1.38, 1.1, 1.4, -0.02, 1.45), 0.75619, 5)

    plt.figure()
    # x = np.linspace(1e-3, 2-1e-3, 400)
    # y = [alpha_mu(xi, 7./4., 5, -1, a=2) for xi in x]

    plt.plot(x, y)
    plt.savefig("alpha-mu.pdf")

def test_alpha_mu_moments():
    pass
