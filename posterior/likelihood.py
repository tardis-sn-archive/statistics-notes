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
    # res = a
    # for i in range(1, b):
    #     res *= a + i
    # return res


def log_poisson_posterior_predictive(N, n):
    '''
    Poisson posterior predictive probability distribution to observe ``N`` given ``n`` on the log scale.

    A Jeffrey's prior is assumed for the rate parameter. The derivation is given in arXiv:1112.2593
    '''
    return log_pochhammer(N + n + 1, N + n) - log_pochhammer(n + 1, n) - (3 * N + n + 0.5) * math.log(2) - gammaln(N)


class TARDISBayesianLogLikelihood(Model):
    inputs = ('packet_nu', 'packet_energies')
    outputs = ('loglikelihood')

    def __init__(self, wavelength, flux):
        super(TARDISBayesianLogLikelihood, self).__init__()
        self.wavelength = wavelength
        self.flux = flux

    def evaluate(self, packet_nu, packet_energies):
         return 5
