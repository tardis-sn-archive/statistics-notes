reload("PaperShortPlotUtils")
plt = PaperShortPlotUtils
plt.plot_λvariable()
plt.plot_convolution()

# just to see sample mean and variance printed out
# reload("TardisPlotUtils")
# TardisPlotUtils.plot_tardis_samples()

E(N, lbar) = (N+1/2)*lbar
V(N,K, l2bar, sig2bar) = (N+1/2)*((N+3/2)/(K-2)* sig2bar + lbar^2)
V(N, lbar, sig2bar)=V(N,N,lbar, sig2bar)
lbar = 0.025613401131261773; sig2bar = 0.00047841795374907296
