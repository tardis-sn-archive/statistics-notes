include("TardisPlotUtils.jl")

# fig:simple_spectrum
TardisPlotUtils.spectrum_inset()

# fig:asymptotic
res = TardisPlotUtils.compute_all_predictions()
TardisPlotUtils.plot_asymptotic_all(res)

# fig:comp-unc
TardisPlotUtils.prepare_compare_uncertainties()

# fig:tardis
TardisPlotUtils.plot_tardis_samples()
