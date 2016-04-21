import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps


a = np.loadtxt("out.txt")

# modify data to include the origin
a = np.vstack((np.zeros(2), a))

X = a.T[0]
P = a.T[1]

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
plt.title("Mode = %g\nmean x = %g +- %g" % (mode, mean, stddev))

plt.tight_layout()
plt.show()

# Local Variables:
# compile-command: "python3 plot.py"
# End:
