from astropy.modeling import Model, Parameter
import math
from math import log
import numpy as np
import scipy
from scipy.special import gammaln
from scipy.stats import gaussian_kde
from scipy.stats import norm as gaussian

'''
Implement likelihood that estimates uncertainty due to tardis simulation
'''

def log_pochhammer(a, b):
    '''
    Pochhammer symbol (a)_b implemented for integer ``b``.
    '''
    assert a > 0
    assert b >= 0
    return gammaln(a+b) - gammaln(a)


def poisson_posterior_predictive(N, n):
    '''
    Poisson posterior predictive probability distribution to observe ``N`` given ``n`` on the log scale.

    A Jeffrey's prior is assumed for the rate parameter. The derivation is given in arXiv:1112.2593
    '''
    return math.exp(log_pochhammer(N + n + 1, N + n) - log_pochhammer(n + 1, n) \
           - (3 * N + n + 0.5) * math.log(2) - gammaln(N + 1))


def binary_search_min(data, key, low=None, high=None):
    '''
    Find the index of the smallest element in ``data`` that is >= ``key``.
    '''

    # update low and high until element found
    if low is None:
        low = 0
    if high is None:
        high = len(data) - 1

    assert low < high

    # first element is the right one
    if data[0] >= key:
        return low
    if data[-1] < key:
        raise KeyError('Key ' + str(key) + ' exceeds all elements')

    while low < (high - 1):
        mid = (low + high) // 2
        midval = data[mid]
        if midval >= key:
            high = mid
        if midval < key:
            low = mid

    return high


def binary_search_max(data, key, low=None, high=None):
    '''
    Find the index of the first element in ``data`` that is > ``key``.
    '''

    # update low and high until element found
    if low is None:
        low = 0
    if high is None:
        high = len(data) - 1

    assert low < high

    # first element is the right one
    if data[-1] <= key:
        return high
    if data[0] > key:
        raise KeyError('Key ' + str(key) + ' less than all elements')

    while low < (high - 1):
        mid = (low + high) // 2
        midval = data[mid]
        if midval <= key:
            low = mid
        if midval > key:
            high = mid

    return low


def binary_search_min_max(data, minkey, maxkey, low=None, high=None):
    '''
    Find the indices of the smallest element exceeding ``minkey`` and of the
    first element exceeding ``maxkey``.

    :param data:
    :param minkey:
    :param maxkey:
    :param low:
    :param high:
    :return:
    '''
    imin = binary_search_min(data, minkey, low, high)
    imax = binary_search_max(data, maxkey, imin, high)
    return (imin, imax)


def expand_bin(N, imin, imax, K):
    '''
    Expand bin defined by lower index ``imin`` and upper index ``imax``
    to contain ``N`` elements. The indices range from 0 to K.

    Attempt to add elements symmetrically on either side. This stops once a
    boundary is hit. If ``len(data) < N``, less element are in the expanded bin.

    :param K: 1D data
    :param imin:
    :param imax:
    :param N:
    :return: tuple (jmin, jmax)
    '''
    assert imin >= 0
    assert imax >= imin
    assert imax <= K
    assert N >= 0
    assert K >= 0

    # combine less elements if data not long enough
    N = min(N, K)

    # elements already in
    n = imax - imin

    # grow to the left first, try to add rest of needed elements, then
    # repeat on the right. If not enough, add more to the left
    jmultimin = max(0, imin - (N - n) // 2)
    jmultimax = min(K, imax + (N - (imax - jmultimin)))
    missing = N- (jmultimax - jmultimin)
    if missing > 0:
        jmultimin = max(0, jmultimin - missing)

    assert jmultimax - jmultimin == N

    return (jmultimin, jmultimax)


def bin_indices(flux, data):
    '''

    :param flux:
    :param data:
    :return:
    '''
    pass

def prob_L_given_N_empirical(N, lsamples):
    '''
    Approximate probability of luminosity ``L`` from generating sums of ``N`` samples
    from ``lsamples`` and interpolating the probability density through Gaussian kernel
    density estimation.

    :param L:
    :param N:
    :param lsamples:
    :param mean:
    :param sigma:
    :return:
    '''

    # todo move loop to cython/C
    summed_samples = np.empty(len(lsamples) // N, dtype=lsamples.dtype)
    assert len(summed_samples) > 1, "Need more than one sample for KDE"
    for i in range(len(summed_samples)):
        summed_samples[i] = lsamples[i:i+N].sum()

    return gaussian_kde(summed_samples, bw_method='scott')

def prob_L_given_N(N, lsamples, mean, variance, N_critical=10):
    '''
    Redirector to prob_L_given_N_empirical or prob_L_given_N_CLT depending on ``N``.

    '''

    assert N > 0

    if N >= N_critical:
        return gaussian(N * mean, math.sqrt(N * variance)).pdf
    else:
        kde = prob_L_given_N_empirical(N, lsamples)
        return lambda L: kde(L)[0] if np.isscalar(L) else kde(L)


def prob_L_given_theta(L, lsamples, imin, imax, nsum=400, eps=1e-3):
    '''P(L|\theta) estimated from a single tardis simulation
    for a single bin.

    :param L: Luminosity (scalar)
    :param lsamples: Array of packet luminosities from tardis
    :param imin: index of first packet in the frequency bin
    :param imax: index of one-past-the-last element in the frequency bin
    :param nsum: number of samples used for KDE interpolation
    :param eps: Relative precision. The contribution for different possible
           number of packets observed is truncated when its contribution to
           the total probability is estimated to be less than ``eps``.
    :return: probability
    '''

    # number of simulated packets and sum of luminosities in the bin i
    ni = imax - imin
    li = lsamples[imin:imax].sum()

    # obtain multibin of luminosities
    # make it large enough to accurately estimate the distribution
    jmultimin, jmultimax = expand_bin(nsum, imin, imax, len(lsamples))
    lmultibin = lsamples[jmultimin:jmultimax]

    # sample mean and standard deviation
    mean, variance = np.mean(lmultibin), np.var(lmultibin)

    # approximate the sum over N, the number of packets one could have seen:
    # 1. Start with N=n, this should be the biggest single contribution
    # 2. Then expand to left and right, add up terms until negligible
    # 3. If N large enough, use central limit theorem
    prob_N_up = poisson_posterior_predictive(ni, ni)
    prob_N_down = prob_N_up
    res = prob_N_up * prob_L_given_N(ni, lmultibin, mean, variance)(L)
    go_up = True
    go_down = True if ni > 1 else False
    N_up, N_down = ni, ni

    while go_up or go_down:
        # store contribution of this expansion
        contrib = 0

        if go_up:
            # posterior predictive P(N+1|n)/P(N|n) = [2(N+n)+1] / [4 (N+1)]
            prob_N_up *= (2. * (N_up + ni) + 1) / (4 * (N_up + 1))

            N_up += 1
            contrib += prob_N_up * prob_L_given_N(N_up, lsamples, mean, variance)(L)

        if go_down:
            # lowest contribution is from N=1. For N=0 packets, there is
            # definitely no contribution to the luminosity
            if N_down == 2:
                go_down = False
            # posterior predictive ratio P(N+1|n)/P(N|n) = 4N / [2(N+n)-1]
            prob_N_down *= (4. * N_down) / (2 * (N_down + ni) - 1)

            N_down -= 1
            contrib += prob_N_down * prob_L_given_N(N_down, lsamples, mean, variance)(L)

        print("N_up: %d, N_down: %d, res: %g, contrib: %g" %
              (N_up, N_down, res, contrib))

        # compare contribution and quit
        if contrib / res < eps:
            go_down = go_up = False

        assert N_up < 2 * ni

        # always use known contribution
        res += contrib

    return res

def amoroso_log_likelihood(parameters, samples, invert=False):#samples, a, mu, sigma, l):
    '''
    -log(likelihood) if all ``samples`` follow an Amoroso distribution in the
    Lawless parametrization.

    :param parameters: a, mu, sigma, l
    :param samples: the samples

    :return: log(likelihood), gradient
    '''
    a, mu, sigma, l = parameters

    N = len(samples)
    lsqinv = l**2
    # scalar
    res = log(math.fabs(l) / sigma) - gammaln(lsqinv) - mu / (l * sigma) + lsqinv * log(lsqinv)
    res *= N

    # invert the centering if ``a`` is the maximum instead of a minimum
    # vector like
    centered = np.log(a - samples) if invert else np.log(samples - a)

    tmp = np.isnan(centered)
    if tmp.any():
        raise ValueError("Encountered nan in centered samples at" +
                         str(np.where(tmp)[0]))

    v = centered.copy()
    v *= (1.0 - 1.0 / (l * sigma))
    tmp = np.exp(-l / sigma * centered)
    v += (lsqinv * math.exp(l * mu / sigma)) * tmp

    res -= v.sum()

    # invert log(likelihood)
    return -res #, gradient


def amoroso_max_likelihood(samples, initial_guess=None, invert=None):
    '''
    Extract the four parameters of an Amoroso distribution through maximum
    likelihood.

    The usual parametrization is that of Crooks, `Survey of simple
    continuous, univariate probability distributions` (2014). But internally,
    the parametrization due to Lawless is used to improve convergence and the
    values in that parametrization are returned.

    :param samples: 1D samples
    :return: (a, mu, sigma, lambda)
    '''
    # todo compute sample_moments and solve to get first estimate
    if initial_guess is None:
        initial_guess = [samples.max(), 0.5, 0.5, 0.5]

    bounds = [(None, 1.05 * samples.max()),
              (None, None),
              (None, None),
              (0, None)]

    return scipy.optimize.minimize(amoroso_log_likelihood, initial_guess,
                                   args=(samples, invert),
                                   bounds=bounds, options=dict(disp=True))


def alpha_mu_log_likelihood(parameters, samples, invert=False, max=None):
    a, alpha, mu, rhat = parameters
    # print(parameters)

    if a < max or alpha <= 0.0 or mu <= 0 or rhat <= 0:
        return np.inf

    # data-independent part
    res = log(alpha) + mu * log(mu) - alpha * mu * log(rhat) - gammaln(mu)
    res *= len(samples)

    centered = a - samples if invert else samples - a
    tmp = np.isnan(centered)
    if tmp.any():
        raise ValueError("Encountered nan in centered samples at" +
                         str(np.where(tmp)[0]))

    res += (alpha * mu - 1) * np.log(centered).sum()

    res -= mu / math.pow(rhat, alpha) * np.power(centered, alpha).sum()

    return -res


def alpha_mu_max_likelihood(samples, initial_guess=None, invert=None):
    '''
    Extract the four parameters of an alpha-mu distribution through maximum
    likelihood.

    Follow Benevides da Costa, Yacoub and Silveira Santos Filho (2008) and
    add the additional location parameter ``a``.

    :param samples: 1D samples
    :return: (a, alpha, mu, rhat)
    '''
    #
    if initial_guess is None:
        initial_guess = [1.01 * samples.max(), 1.1, 1.03, samples.std()]

    bounds = [(None, 1.05 * samples.max()),
              (0, None),
              (0, None),
              (0, None)]

    return scipy.optimize.minimize(alpha_mu_log_likelihood, initial_guess,
                                   args=(samples, invert, samples.max()),
                                   bounds=bounds,
                                   method="Powell", options=dict(disp=True))



def lawless_to_crooks(a, mu, sigma, l):
    '''Convert the Amoroso parameters from the Lawless to the Crooks
    parametrization.

    :return: (a, theta, alpha, beta)
    '''
    return a, math.exp(mu + sigma / l * math.log(l**2)), 1.0 / l**2, l / sigma


class TARDISBayesianLogLikelihood(Model):
    inputs = ('packet_nu', 'packet_energies')
    outputs = ('loglikelihood')

    def __init__(self, wavelength, flux):
        super(TARDISBayesianLogLikelihood, self).__init__()
        self.wavelength = wavelength
        self.flux = flux

    def evaluate(self, packet_nu, packet_energies):
        # sort by nu
        indices = packet_nu.argsort()
        # todo this creates copies!
        packet_nu = packet_nu[indices]
        packet_energies = packet_energies[indices]

        # log(likelihood)
        ll = 0

        # todo run in parallel?
        # divide into bins

        # todo run in parallel?
        # compute log likelihood for each bin
        for b in bins:
            ll += math.log(prob_L_given_theta(L, packet_nu, packet_energies, b[0], b[1]))

        return ll
