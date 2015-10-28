from likelihood import *
from numpy import testing

def test_log_poisson_posterior_predictive():
    args = ((13, 11), (11, 13), (1, 2))
    res  = ()
    testing.assert_almost_equal(log_poisson_posterior_predictive(13, 11), -2.63578)
