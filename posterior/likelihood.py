from astropy.modeling import Model, Parameter
import math
import numpy as np
from scipy.special import gammaln


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


def log_poisson_posterior_predictive(N, n):
    '''
    Poisson posterior predictive probability distribution to observe ``N`` given ``n`` on the log scale.

    A Jeffrey's prior is assumed for the rate parameter. The derivation is given in arXiv:1112.2593
    '''
    return log_pochhammer(N + n + 1, N + n) - log_pochhammer(n + 1, n) \
           - (3 * N + n + 0.5) * math.log(2) - gammaln(N + 1)


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
    Find the index of the largest element in ``data`` that is <= ``key``.
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


def bin_indices(flux, data):
    '''

    :param flux:
    :param data:
    :return:
    '''
    pass

def prob_L_given_N(L, N, lsamples):
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
    # generate the samples of sums of N
    Nsamples = np.array(len(lsamples) // N, dtype=lsamples.dtype)
    for i in range(len(Nsamples)):
        Nsamples[i] = lsamples[i:i+N].sum()

    

def prob_L_given_theta(L, nu, lsamples, imin, imax, nmultibin=1000):
    '''P(L|\theta) estimated from a single tardis simulation
    for a single bin.

    :param L: Luminosity (scalar)
    :param nu: Array of packet frequencies from tardis
    :param lsamples: Array of packet luminosities from tardis
    :param imin: index of first packet in the frequency bin
    :param imax: index of one-past-the-last element in the frequency bin
    :return: probability on the log scale
    '''

    # number of simulated packets and sum of luminosities in the bin i
    ni = imax - imin
    li = lsamples[imin:imax].sum()

    # obtain multibin of luminosities n
    # make it large enough to accurately estimate the distribution
    # grow to the left first, try to add rest of needed elements, then
    # repeat on the right. If not enough, add more to the left
    nmultibin = min(nmultibin, len(nu))
    jmultimin = max(0, imin - (nmultibin - ni) // 2)
    jmultimax = min(len(nu), imax + (nmultibin - (imax - jmultimin)))
    missing = jmultimax - jmultimin
    if missing > 0:
        jmultimin = max(0, jmultimin - missing)

    assert jmultimax-jmultimin == nmultibin
    lmultibin = lsamples[jmultimin:jmultimax]

    # sample mean and standard deviation
    mean, sigma = np.mean(lmultibin), np.std(lmultibin)

    # approximate the sum over N, the number of packets one could have seen:
    # 1. Start with N=n, this should be the biggest single contribution
    # 2. Then expand to left and right and add up terms until contribution negligible
    # 3. If N large enough, use central limit theorem
    log_probN = log_poisson_posterior_predictive(ni, ni)
    res = log_probN + prob_L_given_N(L, N, lmultibin, mean, sigma)


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

        # divide into bins

        # todo run in parallel?
        # compute log likelihood for each bin
        for b in bins:
            prob_L_given_theta(L, packet_nu, packet_energies, b[0], b[1])

         return 5
