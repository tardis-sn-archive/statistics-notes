# reload("TardisPlotUtils")
using Plots
import TardisPaper.Predict
using Distributions

α=1.5; β=60
μ0 = α/β; σ20 = α/β^2
k=34; data = rand(dist, k);
meanℓ = mean(data); varℓ=var(data; corrected=false)


# Plots.gr()
