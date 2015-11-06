from __future__ import print_function
from mixture import *
import matplotlib.pyplot as plt

samples = np.array([  1.40707615,   1.37031156,   1.42174433,
         1.42874909,   1.44447164,   1.41565455,
         1.41348983,   1.42279213,   1.42637690,
         1.42707482,   1.43412572,   1.42627637,
         1.37345854,   1.43557610,   1.44448078,
         1.37408463,   1.42596153,   1.43517086,
         1.41321387,   1.38457203])

def test_interpolator():
    p = PypmcInterpolator(samples, k=5, iterations=500, verbose=True)
    x = np.linspace(0.9 * samples.min(), 1.1 * samples.max(), 100)
    y = np.exp(p(x))
    print((samples.mean(), samples.var(), samples.std()))
    print(p.mixture.components[0].sigma)


    plt.clf()
    plt.hist(samples, bins=5, normed=True, color='grey', alpha=0.3)
    plt.plot(x, y)
    plt.savefig("mixture_test.pdf")
