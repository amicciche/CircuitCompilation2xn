using SafeTestsets
using QuantumClifford

function doset(descr)
    if length(ARGS) == 0
        return true
    end
    for a in ARGS
        if occursin(lowercase(a), lowercase(descr))
            return true
        end
    end
    return false
end

macro doset(descr)
    quote
        if doset($descr)
            @safetestset $descr begin
                include("test_"*$descr*".jl")
            end
        end
    end
end

println("Starting tests with $(Threads.nthreads()) threads out of `Sys.CPU_THREADS = $(Sys.CPU_THREADS)`...")

@doset "base"
#VERSION >= v"1.9" && @doset "doctests"
#get(ENV,"JET_TEST","")=="true" && @doset "jet"
#VERSION >= v"1.9" && @doset "aqua"
