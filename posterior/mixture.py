'''Create 1D interpolation to samples.'''

import numpy as np
import pypmc

class PypmcInterpolator(object):
    def __init__(self, samples, k=8, **kwargs):
        '''
        Create an object that estimates the probabibility density of ``samples`` using mixture
        of Gaussian components via ``pypmc``. Evaluate the estimated density for a single value
        or an array by calling this object.

        :param samples: iterable of samples
        :param k: Maximum number of compnents in the mixture
        :return:
        '''

        # :param dof:
        #     Degrees of freedom. Choose ``dof <= 0`` for a Gaussian.
        #     Any positive value singles out a Student's t distribution.
        self.vb = pypmc.mix_adapt.variational.GaussianInference(samples, components=k, alpha=np.ones(k))
        self.vb.run(**kwargs)
        self.mixture = self.vb.make_mixture()

    def __call__(self, arg):
        # multi_evaluate expects a 2dim array of floats
        if np.isscalar(arg):
            arg = np.array(arg)
        if arg.ndim == 1:
            arg = arg.reshape((len(arg), 1))

        return self.mixture.multi_evaluate(arg)


