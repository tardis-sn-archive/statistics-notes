import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.stats import poisson
import sys, os

def plot_estimate(runfile):

    a = np.loadtxt(runfile)

    # extract run number from file X0.100000-0.120000_run11.out
    s, _ = os.path.splitext(runfile)
    run = int(s[s.find('run')+3:])

    # modify data to include the origin
#     a = np.vstack((np.zeros(2), a))

    X = a.T[0]
    P = np.nan_to_num(a.T[1])

    # check normalization of pdf using Simpson's rule
    norm = simps(P, X)
    print("Normalization", norm)

    # improve estimates by renormalizing to one
    P /= norm

    mean = simps(P * X, X)
    print("Mean", mean)

    stddev = np.sqrt(simps(P * (X - mean)**2, X))
    print("std. dev", stddev)

    mode = X[np.argmax(P)]
    print("Mode", mode)

    plt.plot(X, P)
    plt.xlabel("$X$")
    plt.ylabel("$P(X)$")

    prefix = s[:s.find('run')]
    replicas = np.loadtxt(prefix + 'replica.out')
    print("Observed %d packets and X = %g in run %d" % (replicas[run,0], replicas[run,1], run))
    print("Mean number of packets in bin", replicas.T[0].mean())
    plt.hist(replicas.T[1], bins=15, normed=True)
    plt.vlines(replicas[run, 1], 0, P.max(), color='red', linestyle='dashed')

    plt.title("mode = %g\nmean x = %g +- %g\nsample mean %g +- %g" % (mode, mean, stddev, replicas.T[1].mean(), replicas.T[1].std()))

    plt.tight_layout()
    # plt.show()
    plt.savefig(s + ".pdf")

    plt.clf()
    npack = replicas.T[0]
    res = plt.hist(replicas.T[0], bins=15, normed=False, alpha=0.3,
                   label="%d runs" % len(npack))
    n = np.arange(min(npack), max(npack)+1)
    pmf = poisson.pmf(n, npack.mean())
    # print(n.sum())
    # print(res)
    plt.plot(n, len(npack) * (res[1][1] - res[1][0]) * pmf, 'ro', label='Poisson')
    #     plt.plot(n, pmf, 'ro', label='Poisson best fit')
    plt.axvline(npack[run], 0.0, 1.0, linestyle='dashed', linewidth=2, color='grey')

    plt.xlabel("#packets in bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(s + "_npackets.pdf")

    # plt.clf()
    # totalnpack = replicas.T[2]
    # plt.hist(totalnpack)
    # plt.savefig(prefix + "totalnpackets.pdf")

if __name__ == '__main__':
    plot_estimate(sys.argv[1])
