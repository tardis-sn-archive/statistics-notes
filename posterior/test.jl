using HDF5
using DataFrames
using Plots;
# gadfly()
pyplot()
# gr()


## h5open("real_tardis_250.h5", "r") do file
##     ## dump(file)
##     file["energies"]
##     nus = file["nus"]
##     print(length(energies[1,:]))
##     histogram(energies[:,run])
##     show()
##     Plots.pdf("testPlots.pdf")
## end

function read_data(runid=9, filename="real_tardis_250.h5")
    h5open(filename, "r") do file
        # read only a single run
        file["nus"][:,runid], file["energies"][:,runid]
    end
end

"""
    filter_positive(x, y)

For two arrays of same length, filter out elements with negative x and
return a data frame with columns (nu, energies)

"""

function filter_positive(nus, energies)
    assert(length(energies) == length(nus))

    mask = (energies .> 0.0)
    DataFrame(nus=nus[mask], energies=energies[mask])

    # can't trip result of zip as a collection
    ## filter((x, y) -> y > 0, zip(nus, energies))

    ## # works but yields a subarray that is not so convenient to work with
    ## mask = (energies .> 0.0)
    ## # iterate over columns => few rows, many columns to be cache friendly
    ## x = Array(typeof(energies[1]), 2, length(mask))
    ## pos = 0
    ## for i in eachindex(mask)
    ##     if mask[i]
    ##         pos += 1
    ##         x[1, pos] = nus[i]
    ##         x[2, pos] = energies[i]
    ##     end
    ## end
    ## sub(x,:, 1:pos)


    # CRITICAL take type of first element, assumed to be scalar!
    ## x = Array(typeof(nus[1]), 2, length(energies))
    ## npos::Int64 = 0
    ## for i in eachindex(energies)
    ##     if energies[i] > 0
    ##         npos += 1
    ##         x[1, npos] = nus[i]
    ##         x[2, npos] = energies[i]
    ##     end
    ## end
    ## assert(npos > 0)
    ## return x[:,1:npos]
end

"""

Sort by frequency, then rescale and flip the energies, then rescale
the frequencies.

"""

function transform_data(frame::DataFrame)
    sort!(frame, cols=:nus)

    # update energies

    # reference to the frame if we index directly
    # things like en /= 2 operate on a copy of the data
    en = frame[:energies]
    maxen = (1.0 + 1e-6) * maximum(en)
    for i in eachindex(en)
        en[i] = 1.0 - en[i] / maxen
    end

    # the last frequency is the largest
    frame[:nus] /= frame[:nus][end]
end

## function log_likelihood(frame::DataFrame, param)
##     @parallel (+) for
## end

# type not strictly needed
## runid::Int64 = 9

# splicing: pass returned tuple into next call as individual arguments
raw_data = read_data()
## packets = filter_positive(raw_data...)
## nus = sub(packets, 1, :)
## energies = sub(packets, 2, :)
## println(size(packets), size(nus), size(energies))

frame = filter_positive(raw_data...)
transform_data(frame)

# Plots works directly with data frames
#histogram(frame, :energies)
## histogram(frame, :energies)
## Plots.pdf("testPlots.pdf")
