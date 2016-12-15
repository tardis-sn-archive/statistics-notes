"""
Setup Plots.jl with pyplot the way Fred likes it
"""
module FredPlots
using LaTeXStrings, Plots

function setup_pyplot()
    font = Plots.font("TeX Gyre Heros")
    myfonts = Dict(:guidefont=>font, :legendfont=>font,
                   :xtickfont=>font, :ytickfont=>font);
    Plots.pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)
end

# Called upon import or using FredPlots, after all code in the module
# is compiled.
function __init__()
    setup_pyplot()
#    println("init called setup_pyplot")
end

end
