'''Create 1D interpolation to samples.'''

import pypmc

class PypmcInterpolator(object):
    def __init__(self, samples, dof=10, k=3):
        '''
        Create an object that estimates the probabibility density of 1D
        ``samples`` using mixture of Gaussian or Student's t components via
        ``pypmc``. Evaluate the estimated density for a single value or an
        array by calling this object.

        :param samples: iterable of 1D samples
        :param dof:
            Degrees of freedom. Choose ``dof <= 0`` for a Gaussian.
            Any positive value singles out a Student's t distribution.

        :param k:
        :return:
        '''
        pass

    def __call__(self, arg):
        pass

