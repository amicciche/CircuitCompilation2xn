using Documenter
using CircuitCompilation2xn

ENV["LINES"] = 80    # for forcing `displaysize(io)` to be big enough
ENV["COLUMNS"] = 80
@testset "Doctests" begin
    DocMeta.setdocmeta!(CircuitCompilation2xn, :DocTestSetup, :(using CircuitCompilation2xn); recursive=true)
    doctest(CircuitCompilation2xn)
end
